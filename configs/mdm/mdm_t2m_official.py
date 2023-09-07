_base_ = ['../_base_/datasets/human_ml3d_bs128.py']

# checkpoint saving
checkpoint_config = dict(interval=1)

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# optimizer
optimizer = dict(type='Adam', lr=1e-4)
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
num_layers = 8
num_heads = 4
dropout = 0.1
cond_mask_prob = 0.1
# model settings
model = dict(
    type='MotionDiffusion',
    model=dict(
        type='MDMTransformer',
        input_feats=input_feats,
        latent_dim=latent_dim,
        ff_size=ff_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        time_embed_dim=time_embed_dim,
        cond_mask_prob=cond_mask_prob,
        guide_scale=2.5,
        clip_version='ViT-B/32',
        use_official_ckpt=True
    ),
    loss_recon=dict(type='MSELoss', loss_weight=1, reduction='none'),
    diffusion_train=dict(
        beta_scheduler='cosine',
        diffusion_steps=1000,
        model_mean_type='start_x',
        model_var_type='fixed_small',
    ),
    diffusion_test=dict(
        beta_scheduler='cosine',
        diffusion_steps=1000,
        model_mean_type='start_x',
        model_var_type='fixed_small',
    ),
    inference_type='ddpm'
)