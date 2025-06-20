_base_ = [
    '../../_base_/models/faster_rcnn_r50_dc5.py',
    '../../_base_/datasets/imagenet_vid_fgfa_style.py',
    '../../_base_/default_runtime.py'
]
model = dict(
    type='FGFABatchMulti',
    motion=dict(
        type='FlowNetSimpleOrigin',
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/flownet.ckpt')),
    aggregator=dict(
        type='EmbedAggregatorBatchMulti', num_convs=1, channels=512, kernel_size=3))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[2, 5])
# runtime settings
total_epochs = 7
evaluation = dict(metric=['bbox'], interval=7)
