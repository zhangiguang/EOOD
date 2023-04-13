# dataset settings
dataset_type = 'DOTADataset'
data_root = '/media/iguang/5AB64D2CB64D0A49/DOTA/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval_rot/Annotations/',
        img_prefix=data_root + 'trainval_rot/JPEGImages/',
        # ann_file=data_root + 'val/VOC2007/Annotations/',
        # img_prefix=data_root + 'val/VOC2007/JPEGImages/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val/VOC2007/Annotations/',
        img_prefix=data_root + 'val/VOC2007/JPEGImages/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'val/VOC2007/Annotations/',
        img_prefix=data_root + 'val/VOC2007/JPEGImages/',
        ann_file=data_root + 'val/VOC2007/JPEGImages/',
        # ann_file=data_root + 'ss_test_no_overlap/',
        # img_prefix=data_root + 'ss_test_no_overlap/',
        pipeline=test_pipeline))
