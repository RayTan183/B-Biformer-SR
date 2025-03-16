import torch.nn.functional as F
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, LayerNorm2d, to_2tuple, trunc_normal_
from .ops.bra_legacy import BiLevelRoutingAttention
from ._common import Attention, AttentionLePE

class SCSEModule(nn.Module):
    def __init__(self, ch, re=16):
        super().__init__()
        self.cSE = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(ch,ch//re,1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(ch//re,ch,1),
                                 nn.Sigmoid())
        self.sSE = nn.Sequential(nn.Conv2d(ch,ch,1),
                                 nn.Sigmoid())
    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

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

class BiFormerBlock(nn.Module):
    """
    Attention + FFN
    """
    def __init__(self, dim, drop_path=0., num_heads=8, n_win=7, 
                 qk_dim=None, qk_scale=None, topk=4, mlp_ratio=4, side_dwconv=5):
        super().__init__()
        qk_dim = qk_dim or dim
        self.norm1 = nn.LayerNorm(dim, eps=1e-6) # important to avoid attention collapsing
        if topk > 0:
            self.attn = BiLevelRoutingAttention(
                dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                qk_scale=qk_scale, topk=topk, side_dwconv=side_dwconv)
        elif topk == -1:
            self.attn = Attention(dim=dim)
        elif topk == -2:
            self.attn = AttentionLePE(dim=dim, side_dwconv=side_dwconv)
        elif topk == 0:
            self.attn = nn.Sequential(Rearrange('n h w c -> n c h w'), # compatiability
                                      nn.Conv2d(dim, dim, 1), # pseudo qkv linear
                                      nn.Conv2d(dim, dim, 5, padding=2, groups=dim), # pseudo attention
                                      nn.Conv2d(dim, dim, 1), # pseudo out linear
                                      Rearrange('n c h w -> n h w c')
                                     )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio*dim)),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_ratio*dim), dim)
                                )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            

    def forward(self, x):
        """
        Args:
            x: NHWC tensor
        Return:
            NHWC tensor
        """
        # attention & mlp
        x = x + self.drop_path(self.attn(self.norm1(x))) # (N, H, W, C)
        x = x + self.drop_path(self.mlp(self.norm2(x))) # (N, H, W, C)
        return x

class BiformerLayer(nn.Module):
    """
    Stack several BiFormer Blocks
    """
    def __init__(self, dim, depth, num_heads, n_win, topk,
                 mlp_ratio=4., drop_path=0., side_dwconv=5):

        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList([
            BiFormerBlock(
                    dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    num_heads=num_heads,
                    n_win=n_win,
                    topk=topk,
                    mlp_ratio=mlp_ratio,
                    side_dwconv=side_dwconv,
                )
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.SCSE = SCSEModule(ch=dim)
    def forward(self, x:torch.Tensor):
        """
        Args:
            x: NCHW tensor
        Return:
            NCHW tensor
        """
        # TODO: use fixed window size instead of fixed number of windows
        x = x.permute(0, 2, 3, 1) # NHWC
        for blk in self.blocks:
            x = blk(x)
        x = x.permute(0, 3, 1, 2) # NCHW
        x=self.SCSE(x)
        x = self.conv(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}"


class B_Biformer_SR(nn.Module):
    """
    Replace WindowAttn-ShiftWindowAttn in Swin-T model with Bi-Level Routing Attention
    """
    def __init__(self, in_chans=3,upscale=2,
                 depth=[2],
                 depth_mid=[2,2,2,2,2],
                 embed_dim=[60],
                 embed_dim_mid=120,
                 num_heads=[3],
                 num_heads_mid=[6,6,6,6,6],
                 drop_path_rate=0., drop_rate=0.,
                 img_range=1.,
                 # before_attn_dwconv=3,
                 mlp_ratio=2,
                 norm_layer=LayerNorm2d,
                 ######## biformer specific ############
                 n_wins=4,
                 topks=[1],
                 topks_mid= 8,
                 side_dwconv:int=5,
                 #######################################
                 ):
        super().__init__()
        out_chans = in_chans
        self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_layers = len(depth)
        self.down_scale=2**(self.num_layers)
        self.upscale=upscale
        self.n_wins=n_wins
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        ############ downsample and upsample layers (patch embeddings) ######################
        self.downsample_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()

        # patch embedding: conv-norm
        self.patch_embed = nn.Sequential(nn.Conv2d(in_chans, embed_dim[0], kernel_size=(3, 3), stride=(1, 1),padding=1),
                             norm_layer(embed_dim[0]))

        for i in range(self.num_layers):
            downsample_layer = nn.Sequential(norm_layer(embed_dim[i]),
                        nn.Conv2d(embed_dim[i], embed_dim_mid, kernel_size=(3, 3), stride=(2, 2),padding=1))
            self.downsample_layers.append(downsample_layer)

        for i in reversed(range(self.num_layers)):
            upsample_layer = nn.Sequential(norm_layer(embed_dim_mid),
                        nn.Upsample(scale_factor=2, mode="nearest"),nn.Conv2d(embed_dim_mid, embed_dim[i], 3, padding=1))
            self.upsample_layers.append(upsample_layer)
            concat_conv =nn.Conv2d(2*embed_dim[i], embed_dim[i], kernel_size=(3, 3), stride=(1, 1),padding=1)
            self.concat_back_dim.append(concat_conv)

        ##########################################################################
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        dp_rates_mid = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth_mid))]

        self.layers_down = nn.ModuleList()
        for i_layer in range(len(depth)):
            layer_down = BiformerLayer(dim=embed_dim[i_layer],
                               depth=depth[i_layer],
                               num_heads=num_heads[i_layer],
                               mlp_ratio=mlp_ratio,
                               drop_path=dp_rates[sum(depth[:i_layer]):sum(depth[:i_layer+1])],
                               ####### biformer specific ########
                               n_win=n_wins, topk=topks[i_layer], side_dwconv=side_dwconv
                               ##################################
                               )
            self.layers_down.append(layer_down)

        self.layers_mid = nn.ModuleList()
        for i_layer in range(len(depth_mid)):
            layer_mid = BiformerLayer(dim=embed_dim_mid,
                                    depth=depth_mid[i_layer],
                                    num_heads=num_heads_mid[i_layer],
                                    mlp_ratio=mlp_ratio,
                                    drop_path=dp_rates_mid[sum(depth_mid[:i_layer]):sum(depth_mid[:i_layer + 1])],
                                    ####### biformer specific ########
                                    n_win=n_wins, topk=topks_mid, side_dwconv=side_dwconv
                                    ##################################
                                    )
            self.layers_mid.append(layer_mid)

        self.layers_up = nn.ModuleList()
        for i_layer in reversed(range(len(depth))):
            layer_up = BiformerLayer(dim=embed_dim[i_layer],
                                  depth=depth[i_layer],
                                  num_heads=num_heads[i_layer],
                                  mlp_ratio=mlp_ratio,
                                  drop_path=dp_rates[sum(depth[:i_layer]):sum(depth[:i_layer + 1])],
                                  ####### biformer specific ########
                                  n_win=n_wins, topk=topks[i_layer], side_dwconv=side_dwconv
                                  ##################################
                                  )
            self.layers_up.append(layer_up)
        ##########################################################################
        self.Upsample = UpsampleOneStep(self.upscale, embed_dim[0],out_chans)
        self.norm = norm_layer(embed_dim_mid)
        self.norm_up= norm_layer(embed_dim[0])

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        img_times=self.down_scale*self.n_wins
        mod_pad_h = (img_times - h % img_times) % img_times
        mod_pad_w = (img_times - w % img_times) % img_times
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    #Stage 1: Small Scale Feature Extraction
    def forward_features(self, x:torch.Tensor):
        x = self.patch_embed(x)
        x_downsample = []
        for index,layer_down in enumerate(self.layers_down):
            x_downsample.append(x)
            x = layer_down(x)
            if index<self.num_layers:
              x=self.downsample_layers[index](x)
        x = self.norm(x)
        return x,x_downsample

    # Stage 2: Large Scale Feature Extraction
    def forward_mid_features(self, x):
        for layer_mid in self.layers_mid:
            x = layer_mid(x)
        x = self.norm(x)
        return x

    #Stage 3: Feature Fusion
    def forward_up_features(self, x,x_downsample):
        for inx,layer_up in enumerate(self.layers_up):
            x = self.upsample_layers[inx](x)
            x = torch.cat([x, x_downsample[self.num_layers-1-inx]], 1)

            x = self.concat_back_dim[inx](x)
            x = layer_up(x)
        x = self.norm_up(x)
        return x

    #Stage 4: Reconstruction
    def reconstruction(self,x):
        x=self.Upsample(x)
        return x

    def forward(self, x:torch.Tensor):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x, x_downsample = self.forward_features(x)
        x=self.forward_mid_features(x)
        x = self.forward_up_features(x,x_downsample)
        x = self.reconstruction(x)
        x = x / self.img_range + self.mean

        return x[:, :, :H*self.upscale, :W*self.upscale]

if __name__ == '__main__':
    input = torch.randn(1, 3, 64, 64)
    print(input.shape)
    model = B_Biformer_SR()
    x = model(input)
    print(x.shape)
