
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class AttentionClassificationHead(nn.Module):

    def __init__(self, in_features_dim = 480, num_heads = 8):
        super(AttentionClassificationHead, self).__init__() 

        self.in_features = in_features_dim

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
            nn.Linear(2*in_features_dim, 1280), # 2*in_features because of pooling concatenation
            nn.ReLU(),
            nn.Linear(1280, 628),
            nn.ReLU(),
            nn.Linear(628, 32),
            nn.ReLU(),
            nn.Linear(32, 2), # 2 for binary classication
        )
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
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
        self.avg_pool = torch.mean(out, dim=1)      # [batch, hidden]
        self.max_pool, _ = torch.max(out, dim=1)    # [batch, hidden]
        self.cls_repr = out[:, 0, :]                # take CLS token before pooling [batch, hidden]

        # concatenate
        concat_out = torch.cat((self.avg_pool, self.max_pool), dim=1)  # [batch, 2*hidden]
        self.concat_repr = concat_out

        # classifier
        logits = self.classifier(concat_out)        # [batch, 2]

        return logits
    

# Model = ESM + ClassificationHead
class EsmDeepSec(nn.Module):

    def __init__(self, esm_model, type_head="attention", type_emb_for_classification=""):
        super(EsmDeepSec, self).__init__()

        # Check head type
        types_head = ["attention", "MLP", "CNN"]
        assert type_head in types_head, f"type_head must be one of {types_head}"
        self.type_head = type_head

        # Check emb type
        types_emb_for_classification = ["agg_mean", "agg_max", "cls", "concat(agg_mean, agg_max)", "contextualized_embs"]
        assert type_emb_for_classification in types_emb_for_classification, f"type_emb_for_classification must be one of {type_emb_for_classification}"
        self.type_emb_for_classification = type_emb_for_classification

        # ESM base
        self.esm_model = esm_model  

        # ESM contextualised embeddings dimension (need for input to classifcation head)
        self.ESM_hidden_dim = self.esm_model.config.hidden_size

        # Classification Head
        if type_head == "attention":
            self.feature_fn = AttentionClassificationHead(in_features_dim=self.ESM_hidden_dim)
        elif type_head == "MLP":
            pass
        elif type_head == "CNN":
            pass


    def forward(self, input_ids, attention_mask=None):

        with torch.no_grad(): # Avoid compute gradient on mpdel part that will not be trained
            outputs = self.esm_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            # retrun Dict obj: https://huggingface.co/docs/transformers/en/model_doc/esm#transformers.EsmModel
            # the attributes diepedns on:
                # 1)whcih model has be instationed with AutoModel when ESM was created
                # 2) the cinfug used to initialise ESM model
            #last_hidden_state --> contextalised emb of aa (all tokenes)
            #pooler_output --> emb of the whole sequence (CLS token) AFTER PASSING IN A MLP
            #hidden_states
            #attentions

        self.esm_last_hidden_state = outputs.last_hidden_state # Shape: [batch, seq_len (with special tokens), hidden_dim]

        features = self.feature_fn(self.esm_last_hidden_state) #[batch_size, 2]

        return features

