import torch
import torch.nn as nn
from torch.nn import functional as F

from attenetion import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(32)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x):
        # x: (Batch_size, features, Hight, Width)

        residue = x

        n, c, h, w = x.shape
        # (Batch_size, features, Hight, Width) -> (Batch_size, features, Hight * Width)
        x = x.view(n, c, h * w)

        # (Batch_size, features, Hight * Width) -> (Batch_size, Hight * Width, features)
        x = x.transpose(-1, -2)
        
        # (Batch_size, Hight * Width, features) -> (Batch_size, Hight * Width, features)
        x = self.attention(x)

        # (Batch_size, Hight * Width, features) -> (Batch_size, features, Hight * Width)
        x = x.transpose(-1, -2)

        # (Batch_size, features, Hight * Width) -> (Batch_size, features, Hight, Width)
        x = x.view(n, c, h, w)

        x += residue

        return x
    

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.group_norm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.group_norm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, In_channels, Hight, Width)
        residue = x

        # (Batch_size, In_channels, Hight, Width) -> (Batch_size, In_channels, Hight, Width)
        x = self.group_norm_1(x)

        x = F.silu(x)

        # x: (Batch_size, In_channels, Hight, Width) -> (Batch_size, Out_channels, Hight, Width)
        x = self.conv_1(x)
        
        # (Batch_size, Out_channels, Hight, Width)
        x = self.group_norm_2(x)

        # (Batch_size, Out_channels, Hight, Width)
        x = F.silu(x)
        # (Batch_size, Out_channels, Hight, Width)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)


class VAE_Decoder(nn.Sequential):
    def __init__(self, ):
        super.__init__()
        pass
