import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from torch import autograd as autograd
import mmcv
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from .mmseg.models.segmentors import EncoderDecoder
# from mmseg.models.segmentors import EncoderDecoder
import pywt
import numpy as np
import models.SWT as SWT
# --------------------------------------------
# Perceptual loss
# --------------------------------------------
class EDTEREdgeLocatation(nn.Module):
    def __init__(self, use_input_norm=True, use_range_norm=False):
        super(EDTEREdgeLocatation, self).__init__()
        '''
        use_input_norm: If True, x: [0, 1] --> (x - mean) / std
        use_range_norm: If True, x: [0, 1] --> x: [-1, 1]
        '''
        config_dir=r'/root/autodl-tmp/3DMeasurement/edge_locate/EDTER/configs/bsds/EDTER_BIMLA_320x320_80k_bsds_aug_bs_8.py'
        checkpoint_dir=r'/root/autodl-tmp/3DMeasurement/edge_locate/EDTER/pretrain/EDTER-BSDS-VOC-StageI.pth'
        cfg = mmcv.Config.fromfile(config_dir)
        cfg.model.pretrained = None
        cfg.data.test.test_mode = True
        model = EncoderDecoder(backbone=cfg.model.backbone, decode_head=cfg.model.decode_head,
                               auxiliary_head=cfg.model.auxiliary_head)
        checkpoint = load_checkpoint(model, checkpoint_dir, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = checkpoint['meta']['PALETTE']

        self.use_input_norm = use_input_norm
        self.use_range_norm = use_range_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.FeaturesExtractor =model

    def forward(self, x):
        if self.use_range_norm:
            x = (x + 1.0) / 2.0
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output=self.FeaturesExtractor.whole_inference(img=x, img_meta=None, rescale=None)
        #只取强边缘作为损失强化
        zero = torch.zeros_like(output)
        output = torch.where(output<0.5,zero,output)

        return output


class EdgeAndL1Loss(nn.Module):
    """Edge And L1 loss
    """

    def __init__(self, edge_weights=0.1, lossfn_type='l1', edge_lossfn_type='l1',use_input_norm=True, use_range_norm=False):
        super(EdgeAndL1Loss, self).__init__()
        self.EDTER = EDTEREdgeLocatation(use_input_norm=use_input_norm, use_range_norm=use_range_norm)
        self.lossfn_type = lossfn_type
        self.edge_lossfn_type=edge_lossfn_type
        self.edge_weights = edge_weights
        if self.lossfn_type == 'l1':
            self.lossfn = nn.L1Loss()
        else:
            self.lossfn = nn.MSELoss()

        if self.edge_lossfn_type == 'l1':
            self.edge_lossfn = nn.L1Loss()
        else:
            self.edge_lossfn = nn.MSELoss()

    def forward(self, x, gt):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        l1_loss = 0.0
        l1_loss+=self.lossfn(x,gt)
        x_edter, gt_edter = self.EDTER(x), self.EDTER(gt.detach())
        edge_loss=0.0
        edge_loss +=self.edge_lossfn(x_edter, gt_edter)
        loss =l1_loss+self.edge_weights*edge_loss
        return loss

###############################################################################################
class L1AndSWTLoss(nn.Module):
    def __init__(self, lossfn_type='l1',loss_weight_ll=0.05, loss_weight_lh=0.025, loss_weight_hl=0.025, loss_weight_hh=0.02,
                 reduction='mean'):
        super(L1AndSWTLoss, self).__init__()
        self.loss_weight_ll = loss_weight_ll
        self.loss_weight_lh = loss_weight_lh
        self.loss_weight_hl = loss_weight_hl
        self.loss_weight_hh = loss_weight_hh

        self.criterion = nn.L1Loss(reduction=reduction)
        self.lossfn_type = lossfn_type
        if self.lossfn_type == 'l1':
            self.lossfn = nn.L1Loss()
        else:
            self.lossfn = nn.MSELoss()

    def forward(self, pred, target):
        l1_loss = 0.0
        l1_loss+=self.lossfn(pred,target)
        wavelet = pywt.Wavelet('sym19')

        dlo = wavelet.dec_lo
        an_lo = np.divide(dlo, sum(dlo))
        an_hi = wavelet.dec_hi
        rlo = wavelet.rec_lo
        syn_lo = 2 * np.divide(rlo, sum(rlo))
        syn_hi = wavelet.rec_hi

        filters = pywt.Wavelet('wavelet_normalized', [an_lo, an_hi, syn_lo, syn_hi])
        sfm = SWT.SWTForward(1, filters, 'periodic').to("cuda")

        ## wavelet bands of sr image
        sr_img_y = 16.0 + (pred[:, 0:1, :, :] * 65.481 + pred[:, 1:2, :, :] * 128.553 + pred[:, 2:, :, :] * 24.966)
        # sr_img_cb      = 128 + (-37.797 *pred[:,0:1,:,:] - 74.203 * pred[:,1:2,:,:] + 112.0* pred[:,2:,:,:])
        # sr_img_cr      = 128 + (112.0 *pred[:,0:1,:,:] - 93.786 * pred[:,1:2,:,:] - 18.214 * pred[:,2:,:,:])

        wavelet_sr = sfm(sr_img_y)[0]

        LL_sr = wavelet_sr[:, 0:1, :, :]
        LH_sr = wavelet_sr[:, 1:2, :, :]
        HL_sr = wavelet_sr[:, 2:3, :, :]
        HH_sr = wavelet_sr[:, 3:, :, :]

        ## wavelet bands of hr image
        hr_img_y = 16.0 + (
                    target[:, 0:1, :, :] * 65.481 + target[:, 1:2, :, :] * 128.553 + target[:, 2:, :, :] * 24.966)
        # hr_img_cb      = 128 + (-37.797 *target[:,0:1,:,:] - 74.203 * target[:,1:2,:,:] + 112.0* target[:,2:,:,:])
        # hr_img_cr      = 128 + (112.0 *target[:,0:1,:,:] - 93.786 * target[:,1:2,:,:] - 18.214 * target[:,2:,:,:])

        wavelet_hr = sfm(hr_img_y)[0]

        LL_hr = wavelet_hr[:, 0:1, :, :]
        LH_hr = wavelet_hr[:, 1:2, :, :]
        HL_hr = wavelet_hr[:, 2:3, :, :]
        HH_hr = wavelet_hr[:, 3:, :, :]

        loss_subband_LL = self.loss_weight_ll * self.criterion(LL_sr, LL_hr)
        loss_subband_LH = self.loss_weight_lh * self.criterion(LH_sr, LH_hr)
        loss_subband_HL = self.loss_weight_hl * self.criterion(HL_sr, HL_hr)
        loss_subband_HH = self.loss_weight_hh * self.criterion(HH_sr, HH_hr)

        return l1_loss+loss_subband_LL + loss_subband_LH + loss_subband_HL + loss_subband_HH

###########################################################################################################################
class EdgeAndL1AndSWTLoss(nn.Module):
    """Edge And L1 And SWT loss
    """

    def __init__(self, edge_weights=0.1, lossfn_type='l1', edge_lossfn_type='l1',use_input_norm=True, use_range_norm=False,
                 loss_weight_ll=0.05, loss_weight_lh=0.025, loss_weight_hl=0.025, loss_weight_hh=0.02,reduction='mean'):
        super(EdgeAndL1AndSWTLoss, self).__init__()
        self.EDTER = EDTEREdgeLocatation(use_input_norm=use_input_norm, use_range_norm=use_range_norm)
        self.lossfn_type = lossfn_type
        self.edge_lossfn_type=edge_lossfn_type
        self.edge_weights = edge_weights
        if self.lossfn_type == 'l1':
            self.lossfn = nn.L1Loss()
        else:
            self.lossfn = nn.MSELoss()

        if self.edge_lossfn_type == 'l1':
            self.edge_lossfn = nn.L1Loss()
        else:
            self.edge_lossfn = nn.MSELoss()
        self.loss_weight_ll = loss_weight_ll
        self.loss_weight_lh = loss_weight_lh
        self.loss_weight_hl = loss_weight_hl
        self.loss_weight_hh = loss_weight_hh

        self.criterion = nn.L1Loss(reduction=reduction)
    def forward(self, pred, target):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        l1_loss = 0.0
        l1_loss+=self.lossfn(pred,target)
        x_edter, gt_edter = self.EDTER(pred), self.EDTER(target.detach())
        edge_loss=0.0
        edge_loss +=self.edge_lossfn(x_edter, gt_edter)

        wavelet = pywt.Wavelet('sym19')
        dlo = wavelet.dec_lo
        an_lo = np.divide(dlo, sum(dlo))
        an_hi = wavelet.dec_hi
        rlo = wavelet.rec_lo
        syn_lo = 2 * np.divide(rlo, sum(rlo))
        syn_hi = wavelet.rec_hi

        filters = pywt.Wavelet('wavelet_normalized', [an_lo, an_hi, syn_lo, syn_hi])
        sfm = SWT.SWTForward(1, filters, 'periodic').to("cuda")

        ## wavelet bands of sr image
        sr_img_y = 16.0 + (pred[:, 0:1, :, :] * 65.481 + pred[:, 1:2, :, :] * 128.553 + pred[:, 2:, :, :] * 24.966)
        # sr_img_cb      = 128 + (-37.797 *pred[:,0:1,:,:] - 74.203 * pred[:,1:2,:,:] + 112.0* pred[:,2:,:,:])
        # sr_img_cr      = 128 + (112.0 *pred[:,0:1,:,:] - 93.786 * pred[:,1:2,:,:] - 18.214 * pred[:,2:,:,:])

        wavelet_sr = sfm(sr_img_y)[0]

        LL_sr = wavelet_sr[:, 0:1, :, :]
        LH_sr = wavelet_sr[:, 1:2, :, :]
        HL_sr = wavelet_sr[:, 2:3, :, :]
        HH_sr = wavelet_sr[:, 3:, :, :]

        ## wavelet bands of hr image
        hr_img_y = 16.0 + (
                    target[:, 0:1, :, :] * 65.481 + target[:, 1:2, :, :] * 128.553 + target[:, 2:, :, :] * 24.966)
        # hr_img_cb      = 128 + (-37.797 *target[:,0:1,:,:] - 74.203 * target[:,1:2,:,:] + 112.0* target[:,2:,:,:])
        # hr_img_cr      = 128 + (112.0 *target[:,0:1,:,:] - 93.786 * target[:,1:2,:,:] - 18.214 * target[:,2:,:,:])

        wavelet_hr = sfm(hr_img_y)[0]

        LL_hr = wavelet_hr[:, 0:1, :, :]
        LH_hr = wavelet_hr[:, 1:2, :, :]
        HL_hr = wavelet_hr[:, 2:3, :, :]
        HH_hr = wavelet_hr[:, 3:, :, :]

        loss_subband_LL = self.loss_weight_ll * self.criterion(LL_sr, LL_hr)
        loss_subband_LH = self.loss_weight_lh * self.criterion(LH_sr, LH_hr)
        loss_subband_HL = self.loss_weight_hl * self.criterion(HL_sr, HL_hr)
        loss_subband_HH = self.loss_weight_hh * self.criterion(HH_sr, HH_hr)
        loss =l1_loss+self.edge_weights*edge_loss+loss_subband_LL+loss_subband_LH+loss_subband_HL+loss_subband_HH
        return loss
