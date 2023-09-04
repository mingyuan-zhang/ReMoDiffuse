# dataset settings
data_keys = ['motion', 'motion_mask', 'motion_length', 'clip_feat']
meta_keys = ['text', 'token']
train_pipeline = [
    dict(
        type='Normalize',
        mean_path='data/datasets/human_ml3d/mean.npy',
        std_path='data/datasets/human_ml3d/std.npy'),
    dict(type='Crop', crop_size=196),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect', keys=data_keys, meta_keys=meta_keys)
]

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        dataset=dict(
            type='TextMotionDataset',
            dataset_name='human_ml3d',
            data_prefix='data',
            pipeline=train_pipeline,
            ann_file='train.txt',
            motion_dir='motions',
            text_dir='texts',
            token_dir='tokens',
            clip_feat_dir='clip_feats',
        ),
        times=200
    ),
    test=dict(
        type='TextMotionDataset',
        dataset_name='human_ml3d',
        data_prefix='data',
        pipeline=train_pipeline,
        ann_file='test.txt',
        motion_dir='motions',
        text_dir='texts',
        token_dir='tokens',
        clip_feat_dir='clip_feats',
        eval_cfg=dict(
            shuffle_indexes=True,
            replication_times=20,
            replication_reduction='statistics',
            text_encoder_name='human_ml3d',
            text_encoder_path='data/evaluators/human_ml3d/finest.tar',
            motion_encoder_name='human_ml3d',
            motion_encoder_path='data/evaluators/human_ml3d/finest.tar',
            metrics=[
                dict(type='R Precision', batch_size=32, top_k=3),
                dict(type='Matching Score', batch_size=32),
                dict(type='FID'),
                dict(type='Diversity', num_samples=300),
                dict(type='MultiModality', num_samples=100, num_repeats=30, num_picks=10)
            ]
        ),
        test_mode=True
    )
)