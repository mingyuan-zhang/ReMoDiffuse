_base_ = ['../_base_/datasets/human_ml3d_bs128.py']

# checkpoint saving
checkpoint_config = dict(interval=1)

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# optimizer
optimizer = dict(type='Adam', lr=2e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[])
runner = dict(type='EpochBasedRunner', max_epochs=50)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

input_feats = 263
max_seq_len = 196
latent_dim = 512
time_embed_dim = 2048
text_latent_dim = 256
ff_size = 1024
num_heads = 8
dropout = 0
# model settings
model = dict(
    type='MotionDiffusion',
    model=dict(
        type='MotionDiffuseTransformer',
        input_feats=input_feats,
        max_seq_len=max_seq_len,
        latent_dim=latent_dim,
        time_embed_dim=time_embed_dim,
        num_layers=8,
        sa_block_cfg=dict(
            type='EfficientSelfAttention',
            latent_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            time_embed_dim=time_embed_dim
        ),
        ca_block_cfg=dict(
            type='EfficientCrossAttention',
            latent_dim=latent_dim,
            text_latent_dim=text_latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            time_embed_dim=time_embed_dim
        ),
        ffn_cfg=dict(
            latent_dim=latent_dim,
            ffn_dim=ff_size,
            dropout=dropout,
            time_embed_dim=time_embed_dim
        ),
        text_encoder=dict(
            pretrained_model='clip',
            latent_dim=text_latent_dim,
            num_layers=4,
            num_heads=4,
            ff_size=2048,
            dropout=dropout,
            use_text_proj=True
        )
    ),
    loss_recon=dict(type='MSELoss', loss_weight=1, reduction='none'),
    diffusion_train=dict(
        beta_scheduler='linear',
        diffusion_steps=1000,
        model_mean_type='epsilon',
        model_var_type='fixed_small',
    ),
    diffusion_test=dict(
        beta_scheduler='linear',
        diffusion_steps=1000,
        model_mean_type='epsilon',
        model_var_type='fixed_small',
    ),
    inference_type='ddpm'
)
data = dict(samples_per_gpu=128)