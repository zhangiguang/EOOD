# dataset settings
dataset_type = 'DIORRDataset'
# data_root = '/data2/dailh/dota1-1024-ms/'
# data_root = '/data2/dailh/split_1024_dota1_0/'
data_root = '/media/iguang/5AB64D2CB64D0A49/DIOR/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(800, 800)),
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
        img_scale=(800, 800),
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
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        # ann_file=data_root + 'trainval1024_ms/DOTA_trainval1024_ms.json',
        # ann_file=data_root + 'trainval_rot/Annotations/',
        # img_prefix=data_root + 'trainval_rot/JPEGImages/',
        ann_file=data_root + 'xml_rot/',
        img_prefix=data_root + 'img_rot/',
        # ann_file=data_root + 'val/VOC2007/Annotations/',
        # img_prefix=data_root + 'val/VOC2007/JPEGImages/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/Annotations_test/',
        img_prefix=data_root + 'VOC2007/JPEGImages/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file='/data2/dailh/split_1024_dota1_0/test/' + 'images/',
        # img_prefix='/data2/dailh/split_1024_dota1_0/test/' + 'images/',
        # ann_file=data_root + 'trainval/annfiles/',
        # # img_prefix=data_root + 'trainval/images/',
        ann_file=data_root + 'VOC2007/Annotations_test/',
        img_prefix=data_root + 'VOC2007/JPEGImages/',
        # ann_file=data_root + 'ms_test_png/',
        # img_prefix=data_root + 'ms_test_png/',
        pipeline=test_pipeline))
