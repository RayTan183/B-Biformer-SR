import functools
import torch
from torch.nn import init


"""
# --------------------------------------------
# select the network of G, D and F
# --------------------------------------------
"""


# --------------------------------------------
# Generator, netG, G
# --------------------------------------------
def define_G(opt):
    opt_net = opt['netG']
    net_type = opt_net['net_type']


    # ----------------------------------------
    # denoising task
    # ----------------------------------------

    # ----------------------------------------
    # B-Swinv2-SR
    # ----------------------------------------
    if net_type == 'PHT':
        from models.PHT.PHT import PHT as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_range=opt_net['img_range'],
                   depth=opt_net['depth'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   win_size=opt_net['win_size'],
                   topks=opt_net['topks'],
                   side_dwconv=opt_net['side_dwconv'])
    # TODO
    # ----------------------------------------
    elif net_type == 'B_Biformer_SR':
        from models.B_Biformer_SR.network_B_Biformer_SR import B_Biformer_SR as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   topks=opt_net['topks'],
                   topks_mid=opt_net['topks_mid'],
                   n_wins=opt_net['n_wins'],
                   img_range=opt_net['img_range'],
                   depth=opt_net['depth'],
                   depth_mid= opt_net['depth_mid'],
                   embed_dim=opt_net['embed_dim'],
                   embed_dim_mid = opt_net['embed_dim_mid'],
                   num_heads=opt_net['num_heads'],
                   num_heads_mid = opt_net['num_heads_mid'],
                   mlp_ratio=opt_net['mlp_ratio'])
    # ----------------------------------------
    elif net_type == 'ATD':
        from models.ATD.atd_arch import ATD as net
        netG = net(upscale=opt_net['upscale'],
                   embed_dim=opt_net['embed_dim'],
                   depths=opt_net['depths'],
                   num_heads=opt_net['num_heads'],
                   window_size=opt_net['window_size'],
                   category_size=opt_net['category_size'],
                   num_tokens=opt_net['num_tokens'],
                   reducted_dim=opt_net['reducted_dim'],
                   convffn_kernel_size = opt_net['convffn_kernel_size'],
                   upsampler=opt_net['upsampler'])
    # ----------------------------------------
    elif net_type == 'MambaIR':
        from models.MambaIR.mambair_arch import MambaIR as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   embed_dim=opt_net['embed_dim'],
                   depths=opt_net['depths'],
                   d_state=opt_net['d_state'],
                   mlp_ratio= opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection = opt_net['resi_connection'])
    # ----------------------------------------
    elif net_type == 'BSRGAN':
        from models.BSRGAN.network_rrdbnet import RRDBNet as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nf=opt_net['nf'],
                   nb=opt_net['nb'],
                   gc=opt_net['gc'],
                   sf=opt_net['sf'])
    # ----------------------------------------
    elif net_type == 'HAT':
        from models.HAT.hat_arch import HAT as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   embed_dim=opt_net['embed_dim'],
                   depths=opt_net['depths'],
                   num_heads=opt_net['num_heads'],
                   window_size=opt_net['window_size'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])
    # ----------------------------------------
    elif net_type == 'RCAN':
        from models.RCAN.rcan_arch import RCAN as net
        netG = net(upscale=opt_net['upscale'],
                   num_in_ch=opt_net['num_in_ch'],
                   num_out_ch=opt_net['num_out_ch'],
                   num_feat=opt_net['num_feat'],
                   num_group=opt_net['num_group'],
                   num_block=opt_net['num_block'],
                   squeeze_factor=opt_net['squeeze_factor'])
    else:
        raise NotImplementedError('netG [{:s}] is not found.'.format(net_type))

    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    if opt['is_train']:
        init_weights(netG,
                     init_type=opt_net['init_type'],
                     init_bn_type=opt_net['init_bn_type'],
                     gain=opt_net['init_gain'])

    return netG


# --------------------------------------------
# Discriminator, netD, D
# --------------------------------------------
def define_D(opt):
    opt_net = opt['netD']
    net_type = opt_net['net_type']

    # ----------------------------------------
    # discriminator_vgg_96
    # ----------------------------------------
    if net_type == 'discriminator_vgg_96':
        from models.network_discriminator import Discriminator_VGG_96 as discriminator
        netD = discriminator(in_nc=opt_net['in_nc'],
                             base_nc=opt_net['base_nc'],
                             ac_type=opt_net['act_mode'])

    # ----------------------------------------
    # discriminator_vgg_128
    # ----------------------------------------
    elif net_type == 'discriminator_vgg_128':
        from models.network_discriminator import Discriminator_VGG_128 as discriminator
        netD = discriminator(in_nc=opt_net['in_nc'],
                             base_nc=opt_net['base_nc'],
                             ac_type=opt_net['act_mode'])

    # ----------------------------------------
    # discriminator_vgg_192
    # ----------------------------------------
    elif net_type == 'discriminator_vgg_192':
        from models.network_discriminator import Discriminator_VGG_192 as discriminator
        netD = discriminator(in_nc=opt_net['in_nc'],
                             base_nc=opt_net['base_nc'],
                             ac_type=opt_net['act_mode'])

    # ----------------------------------------
    # discriminator_vgg_128_SN
    # ----------------------------------------
    elif net_type == 'discriminator_vgg_128_SN':
        from models.network_discriminator import Discriminator_VGG_128_SN as discriminator
        netD = discriminator()

    elif net_type == 'discriminator_patchgan':
        from models.network_discriminator import Discriminator_PatchGAN as discriminator
        netD = discriminator(input_nc=opt_net['in_nc'],
                             ndf=opt_net['base_nc'],
                             n_layers=opt_net['n_layers'],
                             norm_type=opt_net['norm_type'])

    elif net_type == 'discriminator_unet':
        from models.network_discriminator import Discriminator_UNet as discriminator
        netD = discriminator(input_nc=opt_net['in_nc'],
                             ndf=opt_net['base_nc'])

    else:
        raise NotImplementedError('netD [{:s}] is not found.'.format(net_type))

    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    init_weights(netD,
                 init_type=opt_net['init_type'],
                 init_bn_type=opt_net['init_bn_type'],
                 gain=opt_net['init_gain'])

    return netD


# --------------------------------------------
# VGGfeature, netF, F
# --------------------------------------------
def define_F(opt, use_bn=False):
    device = torch.device('cuda' if opt['gpu_ids'] else 'cpu')
    from models.network_feature import VGGFeatureExtractor
    # pytorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = VGGFeatureExtractor(feature_layer=feature_layer,
                               use_bn=use_bn,
                               use_input_norm=True,
                               device=device)
    netF.eval()  # No need to train, but need BP to input
    return netF


"""
# --------------------------------------------
# weights initialization
# --------------------------------------------
"""


def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
    """
    # Kai Zhang, https://github.com/cszn/KAIR
    #
    # Args:
    #   init_type:
    #       default, none: pass init_weights
    #       normal; normal; xavier_normal; xavier_uniform;
    #       kaiming_normal; kaiming_uniform; orthogonal
    #   init_bn_type:
    #       uniform; constant
    #   gain:
    #       0.2
    """

    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__

        if classname.find('Conv') != -1 or classname.find('Linear') != -1:

            if init_type == 'normal':
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'uniform':
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)

            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)

            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)

            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(gain)

            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)

            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))

            if m.bias is not None:
                m.bias.data.zero_()

        elif classname.find('BatchNorm2d') != -1:

            if init_bn_type == 'uniform':  # preferred
                if m.affine:
                    init.uniform_(m.weight.data, 0.1, 1.0)
                    init.constant_(m.bias.data, 0.0)
            elif init_bn_type == 'constant':
                if m.affine:
                    init.constant_(m.weight.data, 1.0)
                    init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))

    if init_type not in ['default', 'none']:
        print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))
        fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
        net.apply(fn)
    else:
        print('Pass this initialization! Initialization was done during network definition!')
