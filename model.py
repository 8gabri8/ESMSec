
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class SecFeature(nn.Module):
    def __init__(self, in_features: int = 480) -> None:
        super(SecFeature, self).__init__()  # 调用父类的初始化方法


        self.in_features = in_features


        # 创建注意力层
        self.attention_layer = nn.MultiheadAttention(embed_dim=in_features, num_heads=8)  

        # 创建全连接层
        self.fc_linear1 = nn.Linear(2*in_features, 1280)  
        self.fc_linear2 = nn.Linear(1280, 628)
        self.fc_linear3 = nn.Linear(628, 32)
        self.fc_linear4 = nn.Linear(32, 2)
        # 添加残差连接和归一化层
        self.layer_norm = nn.LayerNorm(in_features)

        self.ffnn = nn.Sequential(
            nn.Linear(in_features, in_features*4),
            nn.GELU(),
            nn.Linear(in_features*4, in_features),
            nn.Dropout(0.3),
        )

     

        self.ffnn_layer_norm = nn.LayerNorm(in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        input_tensor = x.permute(1, 0, 2)

        
        output_tensor, _ = self.attention_layer(input_tensor, input_tensor, input_tensor)
        out = output_tensor.permute(1, 0, 2)
        residual_output = x + out
        out = self.layer_norm(residual_output)
        ffnn_out = self.ffnn(out)

       
        ffnn_output = ffnn_out +out
        out = self.ffnn_layer_norm(ffnn_output)

       
        avg_pool = torch.mean(out, dim=1) 

       
        max_pool, _ = torch.max(out, dim=1)  
        out = torch.cat((avg_pool, max_pool), dim=1)  

       
        out = F.relu(self.fc_linear1(out))
        out = F.relu(self.fc_linear2(out))
        out = F.relu(self.fc_linear3(out))
        out = self.fc_linear4(out)

        return out
