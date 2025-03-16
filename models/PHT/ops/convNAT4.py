"""
Refactored Bi-level Routing Attention that takes NCHW input.

author: ZHU Lei
github: https://github.com/rayleizhu
email: ray.leizhu@outlook.com

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor
from typing import Tuple
from natten import NeighborhoodAttention2D
from einops.layers.torch import Rearrange, Reduce


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


class ConvNat(nn.Module):

    def __init__(self, dim,num_heads=4,expansion_rate=2,shrinkage_rate=0.25):
        super().__init__()
        self.num_heads=num_heads
        self.dim=dim
        self.NAT_attn1 = NeighborhoodAttention2D(dim=dim, kernel_size=31, num_heads=num_heads)
        self.NAT_attn2 = NeighborhoodAttention2D(dim=dim, kernel_size=31, num_heads=num_heads)
        self.dw_conv=nn.Conv2d(dim, dim, 3, stride = 1, padding = 1, groups = dim)
        self.Linear=nn.Linear(dim,dim)

    def forward(self, x:Tensor,x_size):
        """
        Args:
            x: NHWC tensor, better to be channel_last (https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
        Return:
            NHWC tensor
        """
        B, N, C = x.shape
        x=blc_to_bchw(x,x_size)
        x_NAT=self.NAT_attn1(x.permute(0,2,3,1))
        x_NAT = self.NAT_attn2(x_NAT)
        x_conv=self.dw_conv(x)
        output=x_NAT.permute(0,3,1,2)+x_conv
        output=bchw_to_blc(output)
        output=self.Linear(output)
        return output
