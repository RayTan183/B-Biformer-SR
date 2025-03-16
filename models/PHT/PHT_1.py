import torch.nn.functional as F
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, LayerNorm2d, to_2tuple, trunc_normal_
from .ops.swin_bra_ca3 import swin_bra
from ._common import nchwAttentionLePE
from .ops.convNAT3 import ConvNat
# from natten import NeighborhoodAttention2D
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, bias=False,out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.project_in = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(in_features, in_features, kernel_size=1, bias=bias)

    def forward(self, x):

        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

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

class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """
    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)


class Swin_BiFormer_Attention(nn.Module):
    """
    Attention + FFN
    """

    def __init__(self, dim, input_resolution, num_heads=8,win_size=8,
                 qk_scale=None, topk=4, shift_size=0, side_dwconv=5):

        super().__init__()
        self.win_size=win_size
        self.input_resolution = input_resolution
        self.shift_size = shift_size
        if min(self.input_resolution) <= self.win_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-window_size"

        if topk > 0:
            self.attn = swin_bra(dim=dim, shift_size=self.shift_size,num_heads=num_heads, win_size=win_size,
                                qk_scale=qk_scale, topk=topk, side_dwconv=side_dwconv)
        elif topk == -1:
            self.attn = nchwAttentionLePE(dim=dim)
        else:
            raise ValueError('topk should >0 or =-1 !')

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.win_size),
                    slice(-self.win_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.win_size),
                    slice(-self.win_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.win_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.win_size * self.win_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask


    def forward(self, x,x_size):
        """
        Args:
            x: NLC tensor
        Return:
            NLC tensor
        """
        B, L, C = x.shape
        H,W=x_size
        x=x.view(B, H, W, C)
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            #训练阶段大小相同的图像计算注意力
            attn_windows = self.attn(shifted_x, mask=self.attn_mask)  # B H W C
            #测试阶段基于不同大小的测试图计算注意力
        else:
            attn_windows = self.attn(shifted_x, mask=self.calculate_mask(x_size).to(x.device))

        return attn_windows

class Swin_BiFormerBlock(nn.Module):
    """
    Attention + FFN
    """

    def __init__(self, dim, input_resolution, drop_path=0., num_heads=8,
                 win_size=8,act_layer=nn.GELU, qk_scale=None, topk=4, shift_size=0,
                 mlp_ratio=4, side_dwconv=5,drop=0.,norm_layer=nn.LayerNorm):

        super().__init__()
        self.norm1 = norm_layer(dim)  # important to avoid attention collapsing
        self.attn=Swin_BiFormer_Attention(dim=dim, input_resolution=input_resolution,
                                            num_heads=num_heads,win_size=win_size,qk_scale=qk_scale,
                                            topk=topk, shift_size=shift_size, side_dwconv=side_dwconv)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x):
        """
        Args:
            x: NCHW tensor
        Return:
            NCHW tensor
        """
        # attention & mlp
        B, C, H, W = x.size()
        x_size=[H,W]
        x = x.permute(0, 2, 3, 1).flatten(1, 2)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # FFN
        x1 = x + self.drop_path(self.attn(self.norm1(x), x_size))
        x=self.norm2(x1)
        x=x.transpose(1, 2).view(B, C, *x_size)
        x=self.mlp(x)
        x=x.flatten(2).transpose(1, 2)
        x = x1 + self.drop_path(x)  # (N, C, H, W)
        x = x.view(B, H , W, C).permute(0,3,1,2)
        return x

class ConvNatBlock(nn.Module):
    """
    Attention + FFN
    """

    def __init__(self, dim, drop_path=0., num_heads=8, expand_ratio=2, mlp_ratio=4,act_layer=nn.GELU,drop=0.,
                 norm_layer=nn.LayerNorm):

        super().__init__()
        self.norm1 = norm_layer(dim)  # important to avoid attention collapsing

        self.ConvNat=ConvNat(dim=dim,num_heads=num_heads, expansion_rate=expand_ratio,shrinkage_rate=0.25)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: NCHW tensor
        Return:
            NCHW tensor
        """
        # attention & mlp
        B, C, H, W = x.size()
        x_size=[H,W]
        x = x.permute(0, 2, 3, 1).flatten(1, 2)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x1 = x + self.drop_path(self.ConvNat(self.norm1(x),x_size))  # (N, C, H, W)
        x=self.norm2(x1)
        x=x.transpose(1, 2).view(B, C, *x_size)
        x=self.mlp(x)
        x=x.flatten(2).transpose(1, 2)
        x = x1 + self.drop_path(x)  # (N, C, H, W)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        return x

class Swin_BiformerLayer(nn.Module):
    """
    Stack several BiFormer Blocks
    """
    def __init__(self, dim, depth, num_heads, win_size, topk,input_resolution,
                 mlp_ratio=4., drop_path=0., side_dwconv=5):

        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks= nn.ModuleList()
        for i in range(depth):
            if i % 2==0:
                block=Swin_BiFormerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, win_size=win_size,
                                 topk=topk,
                                 mlp_ratio=mlp_ratio,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path)
            else:
                block=ConvNatBlock(dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio)
            self.blocks.append(block)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)



    def forward(self, x:torch.Tensor):
        """
        Args:
            x: NCHW tensor
        Return:
            NCHW tensor
        """
        # TODO: use fixed window size instead of fixed number of windows
        res=x
        for blk in self.blocks:
            x = blk(x)
        x = self.conv(x)
        x = res + x
        return x


class PHT(nn.Module):
    """
    Replace WindowAttn-ShiftWindowAttn in Swin-T model with Bi-Level Routing Attention
    """
    def __init__(self, in_chans=3,upscale=4,
                 depth=[2,2,2,2,2],img_size=64,
                 embed_dim=64,
                 num_heads=8,
                 drop_path_rate=0., drop_rate=0.,
                 img_range=1.,
                 # before_attn_dwconv=3,
                 mlp_ratio=2,
                 norm_layer=LayerNorm2d,
                 ######## biformer specific ############
                 win_size=8,
                 topks=[2,4,8,4,2],
                 side_dwconv:int=5,
                 #######################################
                 ):
        super().__init__()
        out_chans = in_chans
        img_size = to_2tuple(img_size)
        self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_layers = len(depth)
        self.upscale=upscale
        self.win_size=win_size
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        ############ downsample and upsample layers (patch embeddings) ######################

        # patch embedding: conv-norm
        self.patch_embed = nn.Sequential(nn.Conv2d(in_chans, embed_dim, kernel_size=(3, 3), stride=(1, 1),padding=1),
                             norm_layer(embed_dim))


        ##########################################################################
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]

        self.layers = nn.ModuleList()
        for i_layer in range(len(depth)):
            layer = Swin_BiformerLayer(dim=embed_dim,input_resolution=(img_size[0],img_size[1]),
                               depth=depth[i_layer],
                               num_heads=num_heads,
                               mlp_ratio=mlp_ratio,
                               drop_path=dp_rates[sum(depth[:i_layer]):sum(depth[:i_layer+1])],
                               ####### biformer specific ########
                               win_size=win_size, topk=topks[i_layer], side_dwconv=side_dwconv
                               ##################################
                               )
            self.layers.append(layer)

        ##########################################################################
        self.bicubic_upsample = nn.Upsample(scale_factor=self.upscale, mode='bicubic')
        self.Upsample = UpsampleOneStep(self.upscale, embed_dim,out_chans)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        img_times=self.win_size
        mod_pad_h = (img_times - h % img_times) % img_times
        mod_pad_w = (img_times - w % img_times) % img_times
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    #Stage 1: Small Scale Feature Extraction
    def forward_features(self, x:torch.Tensor):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        return x


    #Stage 4: Reconstruction
    def reconstruction(self,x):
        x=self.Upsample(x)
        return x

    def forward(self, x:torch.Tensor):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        x_bicubic=self.bicubic_upsample(x)
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x = self.forward_features(x)
        x = self.reconstruction(x)
        x=x+x_bicubic
        x = x / self.img_range + self.mean

        return x[:, :, :H*self.upscale, :W*self.upscale]

if __name__ == '__main__':
    input = torch.randn(1, 3, 212, 33)
    print(input.shape)
    model = PHT()
    x = model(input)
    print(x.shape)
