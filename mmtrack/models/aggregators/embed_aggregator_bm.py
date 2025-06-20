# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn.bricks import ConvModule
from mmcv.runner import BaseModule

from ..builder import AGGREGATORS


@AGGREGATORS.register_module()
class EmbedAggregatorBatchMulti(BaseModule):
    """Embedding convs to aggregate multi feature maps.

    This module is proposed in "Flow-Guided Feature Aggregation for Video
    Object Detection". `FGFA <https://arxiv.org/abs/1703.10025>`_.

    Args:
        num_convs (int): Number of embedding convs.
        channels (int): Channels of embedding convs. Defaults to 256.
        kernel_size (int): Kernel size of embedding convs, Defaults to 3.
        norm_cfg (dict): Configuration of normlization method after each
            conv. Defaults to None.
        act_cfg (dict): Configuration of activation method after each
            conv. Defaults to dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 num_convs=1,
                 channels=256,
                 kernel_size=3,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(EmbedAggregatorBatchMulti, self).__init__(init_cfg)
        assert num_convs > 0, 'The number of convs must be bigger than 1.'
        self.embed_convs = nn.ModuleList()
        for i in range(num_convs):
            if i == num_convs - 1:
                new_norm_cfg = None
                new_act_cfg = None
            else:
                new_norm_cfg = norm_cfg
                new_act_cfg = act_cfg
            self.embed_convs.append(
                ConvModule(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                    norm_cfg=new_norm_cfg,
                    act_cfg=new_act_cfg))

    def forward(self, x, ref_x):
        """Aggregate reference feature maps `ref_x`.

        The aggregation mainly contains two steps:
        1. Computing the cos similarity between `x` and `ref_x`.
        2. Use the normlized (i.e. softmax) cos similarity to weightedly sum
        `ref_x`.

        Args:
            x (Tensor): of shape [N, C, H, W]
            ref_x (Tensor): of shape [N*num_ref_imgs, C, H, W]. N is the number of reference
                feature maps.

        Returns:
            Tensor: The aggregated feature map with shape [N, C, H, W].
        """
        assert len(ref_x.shape)==4
        # assert len(x.shape) == 4 and len(x) == 1, \
        # "Only support 'batch_size == 1' for x"
        
        # For training:
        # x is (N,C',H',W'), ref_x is (N*ref,C',H',W')
        # For testing:
        # x is (1,C',H',W'), ref_x is (1+2*range,C',H',W') including x as the index 'range'
        batch_size, C, feat_height, feat_width = x.shape
        num_ref_imgs = int(ref_x.shape[0]/x.shape[0])

        x_embed = x
        for embed_conv in self.embed_convs:
            x_embed = embed_conv(x_embed)
        x_embed = x_embed / \
            x_embed.norm(p=2, dim=1, keepdim=True)  # (N,C',H',W')

        ref_x_embed = ref_x
        for embed_conv in self.embed_convs:
            ref_x_embed = embed_conv(ref_x_embed)
        ref_x_embed = ref_x_embed / ref_x_embed.norm(p=2, dim=1, keepdim=True)# (ref*N,C',H',W')/(ref*N,1,H',W') norm2 of vector (n,:,i,j)

        ada_weights = torch.sum(ref_x_embed * x_embed.repeat(num_ref_imgs,1, 1, 1), dim=1, keepdim=True)  # (ref*N,1,H',W') sum up C' channels
        ada_weights = ada_weights.view(batch_size, num_ref_imgs, -1, feat_height, feat_width)  # (N,ref,1,H',W')
        ada_weights = ada_weights.softmax(dim=1)  # (N,ref,1,H',W')
        ada_weights=ada_weights.view(batch_size*num_ref_imgs, -1, feat_height, feat_width) # (ref*N,C',H',W')

        agg_x = ref_x * ada_weights  # (ref*N,C',H',W')
        agg_x = agg_x.view(batch_size, num_ref_imgs,-1, feat_height, feat_width)
        agg_x = torch.sum(agg_x, dim=1, keepdim=True)  # add up all ref features within the batch
        agg_x=agg_x.view(batch_size, -1, feat_height, feat_width)
        return agg_x
