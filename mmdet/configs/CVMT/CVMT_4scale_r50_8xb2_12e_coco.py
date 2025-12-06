from mmengine.config import read_base
from mmengine.model.weight_init import PretrainedInit
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import MultiStepLR
from mmengine.runner.loops import EpochBasedTrainLoop, TestLoop, ValLoop
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.normalization import GroupNorm
from torch.optim.adamw import AdamW
from mmdet.models import (ChannelMapper, DetDataPreprocessor_CVMT,
                          DINOHead, ResNet50_CVMT)
from mmdet.models.losses.focal_loss import FocalLoss
from mmdet.models.losses.iou_loss import GIoULoss
from mmdet.models.losses.smooth_l1_loss import L1Loss
from mmdet.models.task_modules import (BBoxL1Cost, FocalLossCost,
                                       HungarianAssigner, IoUCost)
from mmdet.models.detectors.cvmt import CVMT

with read_base():
    from .._base_.datasets.CVMT_coco_detection import *
    from .._base_.default_runtime import *

model = dict(
    type=CVMT,  # CVMT_base
    num_encoder_queries=150,
    num_fused_queries=150,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type=DetDataPreprocessor_CVMT,
        mean=[100],
        std=[50],
        bgr_to_rgb=False,
        pad_size_divisor=1),
    backbone=dict(
        type=ResNet50_CVMT,
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type=BatchNorm2d, requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
             type=PretrainedInit, checkpoint='torchvision://resnet50')
    ),
    neck=dict(
        type=ChannelMapper,
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type=GroupNorm, num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.0))),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.0)),
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,
        temperature=20),
    bbox_head=dict(
        type=DINOHead,
        num_classes=1,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type=FocalLoss,
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type=L1Loss, loss_weight=5.0),
        loss_iou=dict(type=GIoULoss, loss_weight=2.0)),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),#
    train_cfg=dict(
        assigner=dict(
            type=HungarianAssigner,
            match_costs=[
                dict(type=FocalLossCost, weight=2.0),
                dict(type=BBoxL1Cost, weight=5.0, box_format='xywh'),
                dict(type=IoUCost, iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))

# Optimizer Configuration
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(
        type=AdamW,
        lr=0.0001,
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))

# Learning Strategies
max_epochs = 12
train_cfg = dict(
    type=EpochBasedTrainLoop, max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

param_scheduler = [
    dict(
        type=MultiStepLR,
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.1)
]

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=50),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(
                    type=CheckpointHook,
                    interval=1,
                    max_keep_ckpts=1,
                    save_best='coco/bbox_mAP'
),
    sampler_seed=dict(type=DistSamplerSeedHook),
visualization=dict(type=DetVisualizationHook))


