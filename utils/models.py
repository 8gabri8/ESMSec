
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Union

def get_embs_from_context_embs(
    context_embs_esm,
    attention_mask,
    type_embs = "cls",
    exclude_cls=True
):
    """
    Extracts a sequence-level embedding from the transformer's contextualized
    token embeddings using various pooling and aggregation strategies.

    ATTNETION: Masking is applied for 'agg_mean' and 'agg_max' to ignore padding tokens.

    Returns:
        torch.Tensor: The aggregated embeddings. Shape: [batch_size, final_dim]
                      or [batch_size, seq_len, hidden_size] for 'contextualized_embs'.
    """

    def masked_mean_pooling(embeddings, mask, exclude_cls=True):
        """Computes mean pooling while ignoring padded tokens."""
        
        if exclude_cls:
            # Exclude CLS token (position 0) from aggregation
            embeddings = embeddings[:, 1:, :]
            mask = mask[:, 1:]

        # 1. Expand mask to match embedding dimensions: [B, S, H]
        mask_expanded = mask.unsqueeze(-1).expand_as(embeddings).float()
        
        # 2. Sum the embeddings (numerator: zero out padded token embeddings)
        sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
        
        # 3. Sum the mask (denominator: count of non-padding tokens)
        sum_mask = torch.sum(mask_expanded, dim=1)
        
        # 4. Prevent division by zero (e.g., in case of empty sequences)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        
        # 5. Calculate the masked mean
        return sum_embeddings / sum_mask

    def masked_max_pooling(embeddings, mask, exclude_cls=True):
        """Computes max pooling while ensuring padding tokens are ignored."""

        if exclude_cls:
            # Exclude CLS token (position 0) from aggregation
            embeddings = embeddings[:, 1:, :]
            mask = mask[:, 1:]

        # 1. Expand mask to match embedding dimensions: [B, S, H]
        mask_expanded = mask.unsqueeze(-1).expand_as(embeddings).float()
        
        # 2. Create a temporary copy to modify padding values
        temp_embs = embeddings.clone()
        
        # 3. Set padding positions (where mask is 0) to a very small negative number
        # This ensures they are ignored during the max operation.
        temp_embs[mask_expanded == 0] = -1e9
        
        # 4. Perform max pooling
        max_pool, _ = torch.max(temp_embs, dim=1)
        return max_pool

    # --- Aggregation Logic ---
    
    if type_embs in ["agg_mean", "concat(agg_mean, agg_max)", "concat(agg_mean, agg_max, cls)"]:
        mean_pool = masked_mean_pooling(context_embs_esm, attention_mask, exclude_cls=exclude_cls)
    
    if type_embs in ["agg_max", "concat(agg_mean, agg_max)", "concat(agg_mean, agg_max, cls)"]:
        max_pool = masked_max_pooling(context_embs_esm, attention_mask, exclude_cls=exclude_cls)

    if type_embs == "agg_mean":
        batch_embeddings = mean_pool
        
    elif type_embs == "agg_max":
        batch_embeddings = max_pool
        
    elif type_embs == "cls":
        batch_embeddings = context_embs_esm[:, 0, :] 
        
    elif type_embs == "concat(agg_mean, agg_max)":
        batch_embeddings = torch.cat((mean_pool, max_pool), dim=1) 
        
    elif type_embs == "concat(agg_mean, agg_max, cls)":
        batch_embeddings = torch.cat((mean_pool, max_pool, context_embs_esm[:, 0, :]), dim=1) 
        
    elif type_embs == "contextualized_embs":
        # Returns the full sequence of embeddings for per-token tasks
        batch_embeddings = context_embs_esm
        
    else:
        raise ValueError(f"Unknown embedding type: {type_embs}")

    return batch_embeddings


class MeanPoolMSCNNHead(nn.Module):
    """
    Multi-Scale CNN Head for 1D protein embeddings.
    
    Takes pre-aggregated embeddings (e.g., CLS token, mean/max pooled) and applies
    multi-scale 1D convolutions to extract features at different receptive field sizes.
    
    Architecture:
    1. Reshape 1D embedding into "pseudo-sequence" format
    2. Apply parallel 1D convolutions with different kernel sizes
    3. Global max pooling across each scale
    4. Concatenate multi-scale features
    5. Final MLP classifier
    
    Input: [Batch, Hidden_dim] - e.g., [32, 480]
    Output: [Batch, num_classes] - e.g., [32, 2]
    """
    
    def __init__(self,
                 in_features_dim,                   # Input embedding dimension (e.g., 480)
                 kernel_sizes=[3, 5, 7, 9],        # Different scales for feature extraction
                 num_filters_per_scale=16,          # Filters per scale (increased from 8)
                 num_classes=2,                     # Binary classification
                 dropout_prob=0.3,
                 use_residual=True):                # Add residual connection
        
        super(MeanPoolMSCNNHead, self).__init__()
        
        self.in_features_dim = in_features_dim
        self.kernel_sizes = kernel_sizes
        self.num_scales = len(kernel_sizes)
        self.num_filters = num_filters_per_scale
        self.num_classes = num_classes
        self.use_residual = use_residual
        
        # --- Multi-Scale Convolutional Branches ---
        # Each branch processes the 1D embedding as a "sequence"
        self.conv_branches = nn.ModuleList()
        
        for k_size in kernel_sizes:
            branch = nn.Sequential(
                # Treat embedding as 1 channel with in_features_dim length
                nn.Conv1d(
                    in_channels=1,                    # Single channel
                    out_channels=num_filters_per_scale,
                    kernel_size=k_size,
                    padding=k_size // 2,              # Same padding
                    bias=False                        # BatchNorm will handle bias
                ),
                nn.BatchNorm1d(num_filters_per_scale),
                nn.GELU(),                            # GELU works better than ReLU
                nn.Dropout(dropout_prob * 0.5),       # Light dropout after conv
            )
            self.conv_branches.append(branch)
        
        # --- Feature Fusion ---
        # Total features from all scales
        total_conv_features = self.num_scales * num_filters_per_scale
        
        # Optional: Add residual connection from input
        if use_residual:
            # Project input to same dimension as conv features
            self.residual_proj = nn.Sequential(
                nn.Linear(in_features_dim, total_conv_features),
                nn.LayerNorm(total_conv_features),
                nn.GELU()
            )
            classifier_input_dim = total_conv_features  # After addition
        else:
            self.residual_proj = None
            classifier_input_dim = total_conv_features
        
        # --- Classification MLP ---
        # Deeper network to learn from multi-scale features
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_prob * 0.5),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x, return_embs=False):
        """
        Args:
            x (torch.Tensor): 1D protein embeddings [Batch, Hidden_dim]
                             e.g., [32, 480] from CLS token or mean pooling
            return_embs (bool): Whether to return intermediate embeddings
        
        Returns:
            logits: [Batch, num_classes]
            embs (optional): Dict of intermediate representations
        """
        batch_size = x.shape[0]
        
        # Store original for residual
        x_original = x
        
        # 1. Reshape: [B, H] -> [B, 1, H]
        # Treat the embedding as a 1D sequence with 1 channel
        x = x.unsqueeze(1)  # [Batch, 1, Hidden_dim]
        
        # 2. Apply parallel multi-scale convolutions
        pooled_features_list = []
        
        for conv_branch in self.conv_branches:
            # Conv output: [B, 1, H] -> [B, F, H]
            # where F = num_filters_per_scale
            conv_out = conv_branch(x)
            
            # Global max pooling: [B, F, H] -> [B, F]
            # This extracts the most salient feature from each filter
            pooled_features = F.adaptive_max_pool1d(conv_out, 1).squeeze(2)
            pooled_features_list.append(pooled_features)
        
        # 3. Concatenate all scales: [B, num_scales * F]
        concat_features = torch.cat(pooled_features_list, dim=1)
        
        # 4. Optional residual connection
        if self.use_residual:
            residual_features = self.residual_proj(x_original)
            # Element-wise addition
            concat_features = concat_features + residual_features
        
        # 5. Final classification
        logits = self.classifier(concat_features)
        
        # 6. Return embeddings if requested
        if return_embs:
            embs = {
                "mscnn_1d_concat_features": concat_features,  # [B, num_scales * F]
                "mscnn_1d_per_scale": pooled_features_list,  # List of [B, F] per scale
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
                "class_head_attention_mean": avg_pool.detach(), #shares the same data but is disconnected from the computation graph.
                "class_head_attention_max": max_pool.detach(),
                "class_head_attention_cls": cls_repr.detach(),
            } 
            return logits, embs

        return logits

    
class LogisticRegressionHead(nn.Module):
    """
    Simple logistic regression head: just a single linear layer.
    No hidden layers, no non-linearities - the simplest possible classifier.
    """
    def __init__(self, in_features_dim=320):
        super(LogisticRegressionHead, self).__init__()
        
        # Single linear layer: input -> 2 classes
        self.classifier = nn.Linear(in_features_dim, 2)
    
    def forward(self, x, return_embs=False):
        """
        Args:
            x: Input embeddings [batch_size, in_features_dim]
            return_embs: If True, return logits and embeddings
        
        Returns:
            logits: [batch_size, 2] if return_embs=False
            (logits, embs): tuple if return_embs=True
        """
        logits = self.classifier(x)
        
        if return_embs:
            # Return input embeddings as the "embeddings"
            embs = {'input_features': x}
            return logits, embs
        
        return logits
    

class MLPHead(nn.Module):
    def __init__(self, in_features_dim=480, dropout_prob=0.5):
        super(MLPHead, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_features_dim, 1280),
            nn.LayerNorm(1280),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(1280, 628),
            nn.LayerNorm(628),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(628, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )
        
        self.classifier = nn.Linear(32, 2)

    def forward(self, x, return_embs=False):
        if return_embs:
            h1 = self.layer1(x)
            h2 = self.layer2(h1)
            h3 = self.layer3(h2)
            logits = self.classifier(h3)
            
            embs = {
                'mlp_layer3': h3   # [batch, 32] - most useful for visualization
            }
            return logits, embs
        
        # Normal forward pass
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.classifier(x)


# Model = ESM + ClassificationHead
class EsmDeepSec(nn.Module):

    def __init__(self, esm_model=None, type_head="attention", type_emb_for_classification="contextualized_embs", from_precomputed_embs=False, precomputed_embs_dim=None):
        super(EsmDeepSec, self).__init__()

        # Check head type
        types_head = ["attention", "MLP", "CNN", "LR"]
        assert type_head in types_head, f"type_head must be one of {types_head}"
        self.type_head = type_head

        self.from_precomputed_embs = from_precomputed_embs

        self.esm_model = esm_model  
        self.ESM_hidden_dim = precomputed_embs_dim

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
            self.type_emb_for_classification = type_emb_for_classification # still save the method used to caculte the precompued embs


        # Classification Head
        if type_head == "attention":
            self.class_head = AttentionClassificationHead(in_features_dim=self.in_features_dim)
        elif type_head == "MLP":
            self.class_head = MLPHead(in_features_dim=self.in_features_dim)
        elif type_head == "LR":
            self.class_head = LogisticRegressionHead(in_features_dim=self.in_features_dim)
        elif type_head == "CNN":
            self.class_head = MeanPoolMSCNNHead(in_features_dim=self.in_features_dim)           


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

            input_class_head = get_embs_from_context_embs(
                    context_embs_esm = outputs_esm.last_hidden_state,
                    attention_mask=attention_mask,
                    type_embs = self.type_emb_for_classification,
                    exclude_cls=True
                )

        else:
            input_class_head = precomputed_embs

        # return ONLY embs   (embs of calss_head too)        
        if return_embs:

            embs = {}

            # For sure calculare head embs
            logits, class_head_emb = self.class_head(input_class_head, return_embs=return_embs) # [batch_size, 2]
            embs["class_head_embs"] =  class_head_emb
            
            # Add ESM embeddings ONLY if the ESM model was used
            if outputs_esm is not None:
                embs["esm_mean"] = torch.mean(outputs_esm.last_hidden_state, dim=1)
                embs["esm_max"] = torch.max(outputs_esm.last_hidden_state, dim=1)[0]
                embs["esm_cls"] = outputs_esm.last_hidden_state[:, 0, :]

            # Add precomputed emb in case
            embs["precomputed_embs"] = precomputed_embs
                
            return logits, embs

        # pass thugh class head
        return self.class_head(input_class_head) # [batch_size, 2]

