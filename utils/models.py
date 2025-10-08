
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Union

class MeanPoolMSCNNHead(nn.Module):
    """
    Processes the input embeddings (L x H) by:
    1. Mean Pooling across the Hidden Dimension (H -> 1).
    2. Applying parallel 1D convolutions (MSCNN) along the Sequence Length (L).
    3. Global Max Pooling.
    4. Final Classification MLP.
    """
    def __init__(self,
                 in_features_dim,                  # Original ESM hidden_dim (needed for initial mean pool)
                 kernel_sizes = [3, 5, 7],         # Scales (ie kernel dimensions)
                 num_filters_per_scale = 8,        # Number of filters PER SCALE (like having one filter wiht multiple channels)
                 num_classes = 2,                  # Final output dim
                 dropout_prob = 0.3):
        
        super(MeanPoolMSCNNHead, self).__init__()

        self.in_features_dim = in_features_dim
        self.kernel_sizes = kernel_sizes
        self.num_scales = len(kernel_sizes)
        self.num_filters = num_filters_per_scale
        self.num_classes = num_classes

        # The input channel dimension to the Conv1D layers will be 1 
        # because of the mean pooling (H -> 1).
        cnn_in_channels = 1 

        # --- Multi-Scale Convolutional Branches ---
        self.conv_branches = nn.ModuleList()
        for k_size in kernel_sizes:
            # The Conv1D takes 1 input channel (the mean-pooled value)
            branch = nn.Sequential(
                nn.Conv1d(in_channels=cnn_in_channels,
                          out_channels=num_filters_per_scale,
                          kernel_size=k_size,
                          padding=k_size // 2), # 'same' padding
                nn.GELU(),
                nn.BatchNorm1d(num_filters_per_scale),
                nn.Dropout(dropout_prob),
            )
            self.conv_branches.append(branch)

        # --- Classifier Block ---
        # Input dimension is sum of pooled features: num_scales * num_filters_per_scale
        classifier_input_dim = self.num_scales * num_filters_per_scale
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 32), # Reduced hidden size due to small filter count
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(32, num_classes)
        )

    def forward(self, x, return_embs = False):
        """
        Args:
            x (torch.Tensor): Contextualized embeddings [B, L, H].
        """
        
        # 1. Mean Pooling (H -> 1): [B, L, H] -> [B, L, 1]
        # This collapses the hidden dimension, retaining only the mean feature value per residue.
        mean_pooled_x = torch.mean(x, dim=2, keepdim=True) 

        # 2. Permute input for Conv1d: [B, L, 1] -> [B, 1, L]
        # 1 is the channel dimension, L is the length dimension.
        x_permuted = mean_pooled_x.permute(0, 2, 1)

        pooled_features_list = []
        
        # 3. Process through parallel branches and pool
        for conv_branch in self.conv_branches:
            
            # Conv output: [B, 1, L] -> [B, F, L]
            conv_out = conv_branch(x_permuted) 
            
            # Global Max Pooling: [B, F, L] -> [B, F]
            pooled_features = F.adaptive_max_pool1d(conv_out, 1).squeeze(2)
            pooled_features_list.append(pooled_features)

        # 4. Concatenate pooled features: [B, S * F]
        concat_pooled_features = torch.cat(pooled_features_list, dim=1) 
        
        # 5. Classify
        logits = self.classifier(concat_pooled_features)

        # 6. Handle return_embs
        if return_embs:
            embs = {
                "mscnn_mean_pooled_signal": mean_pooled_x.squeeze(2), # The 1D signal (B x L)
                "mscnn_concat_features": concat_pooled_features,     # The final feature vector (B x S*F)
            } 
            return logits, embs
        
        return logits


class AttentionClassificationHead(nn.Module):

    def __init__(self, in_features_dim = 480, num_heads = 8):
        super(AttentionClassificationHead, self).__init__() 

        self.in_features_dim = in_features_dim

        assert in_features_dim % num_heads == 0, "Number of attnetion heads must be multiple of the hidden dimension."

        # initialise multi-head attention layer
        self.attention_layer = nn.MultiheadAttention(embed_dim=in_features_dim, num_heads=num_heads)  

        # initialize layer normalization layers
        self.layer_norm = nn.LayerNorm(in_features_dim)

        # initialize feed-forward neural network
        self.ffnn = nn.Sequential(
            nn.Linear(in_features_dim, in_features_dim*4),
            nn.GELU(),
            nn.Linear(in_features_dim*4, in_features_dim),
            nn.Dropout(0.3),
        )

        # initialize layer normalization for ffnn
        self.ffnn_layer_norm = nn.LayerNorm(in_features_dim)

        # initialize classification layers
        self.classifier = nn.Sequential(
            nn.Linear(2*in_features_dim, 1280), # 2*in_features_dim because of pooling concatenation
            nn.ReLU(),
            nn.Linear(1280, 628),
            nn.ReLU(),
            nn.Linear(628, 32),
            nn.ReLU(),
            nn.Linear(32, 2), # 2 for binary classication
        )
    

    def forward(self, x, return_embs=False):
        
        # permute: [batch_size, seq_len, hidden_dim] --> [seq_len, batch, hidden_dim] (needed by attention layer)
        input_tensor = x.permute(1, 0, 2)

        # compute SELF attention (Q,K,V are all the same)
        output_tensor, _ = self.attention_layer(input_tensor, input_tensor, input_tensor)

        # return to original shape
        out = output_tensor.permute(1, 0, 2)

        # Skip connections (the block only needs to learn the residual change to apply, not totally new rep)
        residual_output = x + out
        # layer normalization: center the emb of each residue using the values of THAT residue
        out = self.layer_norm(residual_output)

        # Run FFNN (mix info PER token)
        ffnn_out = self.ffnn(out)

        # Skip connection + layer normalization
        ffnn_output = ffnn_out + out
        out = self.ffnn_layer_norm(ffnn_output)

        # store emb to access later if needed
        avg_pool = torch.mean(out, dim=1)      # [batch, hidden]
        max_pool, _ = torch.max(out, dim=1)    # [batch, hidden]
        cls_repr = out[:, 0, :]                # take CLS token before pooling [batch, hidden]

        # concatenate
        concat_out = torch.cat((avg_pool, max_pool), dim=1)  # [batch, 2*hidden]

        # classifier
        logits = self.classifier(concat_out)        # [batch, 2]

        if return_embs:
            logits = logits
            embs = {
                "class_head_mean": avg_pool.detach(), #shares the same data but is disconnected from the computation graph.
                "class_head_max": max_pool.detach(),
                "class_head_cls": cls_repr.detach(),
            } 
            return logits, embs

        return logits
    

class MLPHead(nn.Module):
    def __init__(self, in_features_dim=480, dropout_prob=0.3):
        super(MLPHead, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_features_dim, 1280),
            nn.LayerNorm(1280),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(1280, 628),
            nn.LayerNorm(628),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(628, 32),
            nn.LayerNorm(32),
            nn.ReLU(),

            nn.Linear(32, 2)  # Binary classification
        )

    def forward(self, x, return_embs=False):

        if return_embs:
            logits = self.mlp(x)
            embs = {} # no internal interesting embeddings to report
            return logits, embs
        
        return self.mlp(x) # logits [batch, 2]



# Model = ESM + ClassificationHead
class EsmDeepSec(nn.Module):

    def __init__(self, esm_model, type_head="attention", type_emb_for_classification="contextualized_embs", from_precomputed_embs=False, precomputed_embs_dim=None):
        super(EsmDeepSec, self).__init__()

        # Check head type
        types_head = ["attention", "MLP", "CNN"]
        assert type_head in types_head, f"type_head must be one of {types_head}"
        self.type_head = type_head

        self.from_precomputed_embs = from_precomputed_embs

        # Checj if ESM is needed
        if not self.from_precomputed_embs:

            # Check emb type
            types_emb_for_classification = {
                "1D": ["agg_mean", "agg_max", "cls", "concat(agg_mean, agg_max)", "concat(agg_mean, agg_max, cls)"], # for: MLP, CNN
                "2D": ["contextualized_embs"] # for: attention
            }
            valid_emb_types = [v for values in types_emb_for_classification.values() for v in values]
            assert type_emb_for_classification in valid_emb_types, f"type_emb_for_classification must be one of {list(valid_emb_types)}"        
            self.type_emb_for_classification = type_emb_for_classification

            # ESM base
            self.esm_model = esm_model  

            # ESM contextualised embeddings dimension 
            self.ESM_hidden_dim = self.esm_model.config.hidden_size

            # define in_feature_dim for classifcation head
            if type_emb_for_classification == "concat(agg_mean, agg_max)":
                self.in_features_dim = 2 * self.ESM_hidden_dim #concatenation
            elif type_emb_for_classification == "concat(agg_mean, agg_max, cls)":
                self.in_features_dim = 3 * self.ESM_hidden_dim #concatenation
            else:
                self.in_features_dim = self.ESM_hidden_dim
        
        else: 
            assert precomputed_embs_dim is not None, "precomputed_embs_dim must be provided when from_precomputed_embs=True."
            self.in_features_dim = precomputed_embs_dim
            self.from_precomputed_embs = from_precomputed_embs

        # Classification Head
        if type_head == "attention":
            self.class_head = AttentionClassificationHead(in_features_dim=self.in_features_dim)
        elif type_head == "MLP":
            self.class_head = MLPHead(in_features_dim=self.in_features_dim)
        elif type_head == "CNN":
            pass            


    def forward(self, input_ids=None, attention_mask=None, return_embs=False, precomputed_embs=None):

        # define in case we are not using due to precimpued embs
        outputs_esm = None

        if self.from_precomputed_embs:
            assert precomputed_embs is not None, "precomputed_embs must be provided when from_precomputed_embs=True."

        if not self.from_precomputed_embs:

            with torch.no_grad(): # Avoid compute gradient on mpdel part that will not be trained

                outputs_esm = self.esm_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                # retrun Dict obj: https://huggingface.co/docs/transformers/en/model_doc/esm#transformers.EsmModel
                # the attributes diepedns on:
                    # 1)whcih model has be instationed with AutoModel when ESM was created
                    # 2) the cinfug used to initialise ESM model
                #last_hidden_state --> contextalised emb of aa (all tokenes)
                #pooler_output --> emb of the whole sequence (CLS token) AFTER PASSING IN A MLP
                #hidden_states --> not jsut the last one
                #attentions

            # Define input to classfication head                   
            if self.type_emb_for_classification == "agg_mean":
                input_class_head = torch.mean(outputs_esm.last_hidden_state, dim=1)
            elif self.type_emb_for_classification == "agg_max":
                input_class_head, _ = torch.max(outputs_esm.last_hidden_state, dim=1)
            elif self.type_emb_for_classification == "cls":
                input_class_head = outputs_esm.last_hidden_state[:, 0, :] 
            elif self.type_emb_for_classification == "concat(agg_mean, agg_max)":
                mean_pool = torch.mean(outputs_esm.last_hidden_state, dim=1)
                max_pool, _ = torch.max(outputs_esm.last_hidden_state, dim=1)
                input_class_head = torch.cat((mean_pool, max_pool), dim=1) 
            elif self.type_emb_for_classification == "concat(agg_mean, agg_max, cls)":
                mean_pool = torch.mean(outputs_esm.last_hidden_state, dim=1)
                max_pool, _ = torch.max(outputs_esm.last_hidden_state, dim=1)
                cls = outputs_esm.last_hidden_state[:, 0, :] 
                input_class_head = torch.cat((mean_pool, max_pool, cls), dim=1) 
            elif self.type_emb_for_classification == "contextualized_embs":
                input_class_head = outputs_esm.last_hidden_state # Shape: [batch, seq_len (with special tokens), hidden_dim]

        else:
            input_class_head = precomputed_embs

        # return ONLY embs   (embs of calss_head too)        
        if return_embs:
            logits, class_head_emb = self.class_head(input_class_head, return_embs=return_embs) # [batch_size, 2]
            
            embs = {"class_head_embs": class_head_emb}
            
            # Add ESM embeddings ONLY if the ESM model was used
            if outputs_esm is not None:
                embs["esm_mean"] = torch.mean(outputs_esm.last_hidden_state, dim=1)
                embs["esm_max"] = torch.max(outputs_esm.last_hidden_state, dim=1)[0]
                embs["esm_csl"] = outputs_esm.last_hidden_state[:, 0, :]
                
            return logits, embs

        # pass thugh class head
        return self.class_head(input_class_head) # [batch_size, 2]

