
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# Classification Head
class AttentionClassificationHead(nn.Module):

    def __init__(self, in_features: int = 480) -> None:
        super(AttentionClassificationHead, self).__init__() 

        self.in_features = in_features

        # initialise multi-head attention layer
        # ATTENTION: in_features % num_heads != 0 → crash.
        self.attention_layer = nn.MultiheadAttention(embed_dim=in_features, num_heads=8)  

        # initialize layer normalization layers
        self.layer_norm = nn.LayerNorm(in_features)

        # initialize feed-forward neural network
        self.ffnn = nn.Sequential(
            nn.Linear(in_features, in_features*4),
            nn.GELU(),
            nn.Linear(in_features*4, in_features),
            nn.Dropout(0.3),
        )

        # initialize layer normalization for ffnn
        self.ffnn_layer_norm = nn.LayerNorm(in_features)

        # initialize classification layers
        self.classifier = nn.Sequential(
            nn.Linear(2*in_features, 1280), # 2*in_features because of pooling concatenation
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
    

# Define Model (ESM+ClassificatioHead)
class EsmDeepSec(nn.Module):

    def __init__(self, esm_model):
        super(EsmDeepSec, self).__init__()

        # ESM base
        self.esm_model = esm_model  

        self.ESM_hidden_dim = self.esm_model.config.hidden_size

        # Classification Head
        self.feature_fn = AttentionClassificationHead(in_features=self.ESM_hidden_dim)


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





from torch.utils.data import Dataset

class ProteinDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels, names):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.names = names  # can be strings

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.labels[idx],
            self.names[idx],
        )
