import torch.nn.functional as F
import torch
import torch.nn as nn
from einops import rearrange

class ChannelAttention(nn.Module):

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, 'dim must be divisible by num_heads!'
        self.head_dim = self.dim // self.num_heads

        self.qkv_linear = nn.Linear(dim, 3*dim, bias=True)
        self.scale = self.dim ** -0.5
        self.attn_act = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)


    def forward(self, x):
        """
        Args:
            x: NLC tensor
        Return:
            NLC tensor
        """
        B, L,C = x.size()
        # B, C, H, W = x.size()
        # L=H*W
        # x=x.permute(0, 2, 3, 1).flatten(1, 2)  #B L C
        qkv = self.qkv_linear(x).unsqueeze(3)       # B L 3*C 1
        q, k, v = qkv.chunk(3, dim=2)  # B L C 1
        q = q.reshape(B, L, self.num_heads, C // self.num_heads,1).permute(0,2,1,3,4)   #B H L C/H 1
        k = k.reshape(B, L, self.num_heads, C // self.num_heads, 1).permute(0,2,1,3,4)  #B H L C/H 1
        v = v.reshape(B, L, self.num_heads, C // self.num_heads, 1).permute(0,2,1,3,4)  #B H L C/H 1
        attn_weight = (q * self.scale) @ k.transpose(-2,-1)
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v #B H L C/H 1
        out=out.squeeze(4).transpose(1, 2).reshape(B, L, C)
        out=self.proj(out)
        # out = rearrange(out, 'n (h w) c -> n h w c', h=H, w=W)

        return out


if __name__ == '__main__':
    input = torch.randn(32, 64, 100, 200)
    print(input.shape)
    model = ChannelAttention(dim=64)
    x = model(input)
    print(x.shape)
