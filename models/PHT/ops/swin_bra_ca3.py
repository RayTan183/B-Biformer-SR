"""
Refactored Bi-level Routing Attention that takes NCHW input.

author: ZHU Lei
github: https://github.com/rayleizhu
email: ray.leizhu@outlook.com

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import List, Optional
# from visualizer import get_local
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor
from typing import Tuple
from .torch.rrsda import regional_routing_attention_torch
from .Channel_Attention import ChannelAttention


def bchw_to_bhwc(x: torch.Tensor) -> torch.Tensor:
    """Permutes a tensor from the shape (B, C, H, W) to (B, H, W, C)."""
    return x.permute(0, 2, 3, 1)


def bhwc_to_bchw(x: torch.Tensor) -> torch.Tensor:
    """Permutes a tensor from the shape (B, H, W, C) to (B, C, H, W)."""
    return x.permute(0, 3, 1, 2)


def bchw_to_blc(x: torch.Tensor) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, C, H, W) to (B, L, C)."""
    return x.flatten(2).transpose(1, 2)


def blc_to_bchw(x: torch.Tensor, x_size: Tuple) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, L, C) to (B, C, H, W)."""
    B, L, C = x.shape
    return x.transpose(1, 2).view(B, C, *x_size)


def blc_to_bhwc(x: torch.Tensor, x_size: Tuple) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, L, C) to (B, H, W, C)."""
    B, L, C = x.shape
    return x.view(B, *x_size, C)

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class swin_bra(nn.Module):
    """Bi-Level Routing Attention that takes nchw input

    Compared to legacy version, this implementation:
    * removes unused args and components
    * uses nchw input format to avoid frequent permutation

    When the size of inputs is not divisible by the region size, there is also a numerical difference
    than legacy implementation, due to:
    * different way to pad the input feature map (padding after linear projection)
    * different pooling behavior (count_include_pad=False)

    Current implementation is more reasonable, hence we do not keep backward numerical compatiability
    """
    def __init__(self, dim, shift_size,num_heads=8, win_size=8, qk_scale=None, topk=4,
                 side_dwconv=3, auto_pad=False, attn_backend='torch'):
        super().__init__()
        # local attention setting
        self.shift_size=shift_size
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, 'dim must be divisible by num_heads!'
        self.head_dim = self.dim // self.num_heads
        self.scale = qk_scale or self.dim ** -0.5 # NOTE: to be consistent with old models.

        ####################Channel attention####################################
        self.ca=ChannelAttention(dim=dim,num_heads=num_heads)

        ################side_dwconv (i.e. LCE in Shunted Transformer)###########
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
                    lambda x: torch.zeros_like(x)
        
        ################ regional routing setting #################
        self.topk = topk
        self.win_size = win_size  # number of windows per row/col

        ##########################################
        self.window_qkv_linear = nn.Linear(dim//2, dim//2 * 3, bias=True)
        self.bra_qkv_linear = nn.Linear(dim // 2, dim // 2 * 3, bias=True)
        self.output_linear = nn.Linear(self.dim, self.dim)
        self.softmax = nn.Softmax(dim=-1)

        if attn_backend == 'torch':
            self.attn_fn = regional_routing_attention_torch
        else:
            raise ValueError('CUDA implementation is not available yet. Please stay tuned.')

    # @get_local('bra_output')

    def forward(self, x:Tensor, mask=None):
        """
        Args:
            x: NHWC tensor, better to be channel_last (https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
        Return:
            NHWC tensor
        """
        N, H,W,C = x.size()
        region_size = (self.win_size, self.win_size)

        # STEP 1: split x to c//2 channels
        window_x, bra_x=torch.split(x, C // 2, dim=3)

        # STEP 2: Window attention
        window_x = window_partition(window_x, self.win_size)  # nW*B, window_size, window_size, C//2
        window_x = window_x.view(-1, self.win_size * self.win_size, C//2)  # nW*B, window_size*window_size, C//2
        Bw, Nw, Cw = window_x.size()
        window_qkv = self.window_qkv_linear(window_x)  # nW*B,window_size*window_size,3C//2
        window_qkv = window_qkv.reshape(Bw, Nw, 3, self.num_heads, (C//2) // self.num_heads).permute(2, 0, 3, 1, 4) # 3,nW*B, nH, window_size*window_size, c
        q1, k1, v1 = window_qkv[0], window_qkv[1], window_qkv[2]  # nW*B, nH, window_size*window_size, c
        lcev1=v1.transpose(1, 2).reshape(Bw, Nw, C//2)
        lcev1=window_reverse(lcev1,self.win_size, H, W).permute(0,3,1,2)
        q1 = q1 * self.scale
        window_attn = (q1 @ k1.transpose(-2, -1))


        if mask is not None:
            nW = mask.shape[0]
            window_attn = window_attn.view(Bw // nW, nW, self.num_heads, Nw, Nw) + mask.unsqueeze(1).unsqueeze(0)
            window_attn = window_attn.view(-1, self.num_heads, Nw, Nw)
            window_attn = self.softmax(window_attn)
            window_output = (window_attn @ v1).transpose(1, 2).reshape(Bw, Nw, C // 2)
            window_output = window_output.view(-1, self.win_size, self.win_size, C // 2 )
            window_output = window_reverse(window_output, self.win_size, H, W)  # B H W C//2
            window_output = torch.roll(window_output, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            window_output = bhwc_to_bchw(window_output)
        else:
            window_attn = self.softmax(window_attn)
            window_output = (window_attn @ v1).transpose(1, 2).reshape(Bw, Nw, C // 2)
            window_output = window_reverse(window_output, self.win_size, H, W)  # B H' W' C
            window_output=bhwc_to_bchw(window_output)

        # STEP 3: region-to-region routing
        # NOTE: ceil_mode=True, count_include_pad=False = auto padding
        # NOTE: gradients backward through token-to-token attention. See Appendix A for the intuition.
        bra_qkv=self.bra_qkv_linear(bra_x)  # B H W 3C//2

        q, k, v = bhwc_to_bchw(bra_qkv).chunk(3, dim=1) # B c//2 H W
        q_r = F.avg_pool2d(q.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)# ncpp
        k_r = F.avg_pool2d(k.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False) # ncpp
        q_r:Tensor = q_r.permute(0, 2, 3, 1).flatten(1, 2) # n(p2)c
        k_r:Tensor = k_r.flatten(2, 3) # nc(p2)
        a_r = q_r @ k_r # n(p2)(p2), adj matrix of regional graph
        _, idx_r = torch.topk(a_r, k=self.topk, dim=-1) # n(hw)k long tensor
        idx_r:LongTensor = idx_r.unsqueeze_(1).expand(-1, self.num_heads, -1, -1) 

        bra_output, attn_mat = self.attn_fn(query=q, key=k, value=v, scale=self.scale,
                                        region_graph=idx_r, region_size=region_size)

        output = torch.cat([window_output, bra_output], dim=1)
        output = output + self.lepe(torch.cat([v,lcev1], dim=1)) # ncHW
        output=bchw_to_blc(output)
        # output = self.output_linear(output)
        output=self.ca(output)

        return output
