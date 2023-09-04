from abc import ABCMeta, abstractmethod
from cv2 import norm
import torch
from torch import layer_norm, nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
import numpy as np

from ..builder import SUBMODULES, build_attention
from .position_encoding import SinusoidalPositionalEncoding, LearnedPositionalEncoding
from ..utils.stylization_block import StylizationBlock
import math
import clip


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class FFN(nn.Module):

    def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb, **kwargs):
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        y = x + self.proj_out(y, emb)
        return y
    

class DecoderLayer(nn.Module):

    def __init__(self,
                 sa_block_cfg=None,
                 ca_block_cfg=None,
                 ffn_cfg=None):
        super().__init__()
        self.sa_block = build_attention(sa_block_cfg)
        self.ca_block = build_attention(ca_block_cfg)
        self.ffn = FFN(**ffn_cfg)

    def forward(self, **kwargs):
        if self.sa_block is not None:
            x = self.sa_block(**kwargs)
            kwargs.update({'x': x})
        if self.ca_block is not None:
            x = self.ca_block(**kwargs)
            kwargs.update({'x': x})
        if self.ffn is not None:
            x = self.ffn(**kwargs)
        return x


class DiffusionTransformer(BaseModule, metaclass=ABCMeta):
    def __init__(self,
                 input_feats,
                 max_seq_len=240,
                 latent_dim=512,
                 time_embed_dim=2048,
                 num_layers=8,
                 sa_block_cfg=None,
                 ca_block_cfg=None,
                 ffn_cfg=None,
                 text_encoder=None,
                 use_cache_for_text=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.input_feats = input_feats
        self.max_seq_len = max_seq_len
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.time_embed_dim = time_embed_dim
        self.sequence_embedding = nn.Parameter(torch.randn(max_seq_len, latent_dim))
        
        self.use_cache_for_text = use_cache_for_text
        if use_cache_for_text:
            self.text_cache = {}
        self.build_text_encoder(text_encoder)

        # Input Embedding
        self.joint_embed = nn.Linear(self.input_feats, self.latent_dim)

        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        self.build_temporal_blocks(sa_block_cfg, ca_block_cfg, ffn_cfg)
        
        # Output Module
        self.out = zero_module(nn.Linear(self.latent_dim, self.input_feats))
        
    def build_temporal_blocks(self, sa_block_cfg, ca_block_cfg, ffn_cfg):
        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.temporal_decoder_blocks.append(
                DecoderLayer(
                    sa_block_cfg=sa_block_cfg,
                    ca_block_cfg=ca_block_cfg,
                    ffn_cfg=ffn_cfg
                )
            )
        
    def build_text_encoder(self, text_encoder):
        
        text_latent_dim = text_encoder['latent_dim']
        num_text_layers = text_encoder.get('num_layers', 0)
        text_ff_size = text_encoder.get('ff_size', 2048)
        pretrained_model = text_encoder['pretrained_model']
        text_num_heads =  text_encoder.get('num_heads', 4)
        dropout = text_encoder.get('dropout', 0)
        activation = text_encoder.get('activation', 'gelu')
        self.use_text_proj = text_encoder.get('use_text_proj', False)

        if pretrained_model == 'clip':
            self.clip, _ = clip.load('ViT-B/32', "cpu")
            set_requires_grad(self.clip, False)
            if text_latent_dim != 512:
                self.text_pre_proj = nn.Linear(512, text_latent_dim)
            else:
                self.text_pre_proj = nn.Identity()
        else:
            raise NotImplementedError()
        
        if num_text_layers > 0:
            self.use_text_finetune = True
            textTransEncoderLayer = nn.TransformerEncoderLayer(
                d_model=text_latent_dim,
                nhead=text_num_heads,
                dim_feedforward=text_ff_size,
                dropout=dropout,
                activation=activation)
            self.textTransEncoder = nn.TransformerEncoder(
                textTransEncoderLayer,
                num_layers=num_text_layers)
        else:
            self.use_text_finetune = False
        self.text_ln = nn.LayerNorm(text_latent_dim)
        if self.use_text_proj:
            self.text_proj = nn.Sequential(
                nn.Linear(text_latent_dim, self.time_embed_dim)
            )
        
    def encode_text(self, text, clip_feat, device):
        B = len(text)
        text = clip.tokenize(text, truncate=True).to(device)
        if clip_feat is None:
            with torch.no_grad():
                x = self.clip.token_embedding(text).type(self.clip.dtype)  # [batch_size, n_ctx, d_model]

                x = x + self.clip.positional_embedding.type(self.clip.dtype)
                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.clip.transformer(x)
                x = self.clip.ln_final(x).type(self.clip.dtype)
        else:
            x = clip_feat.type(self.clip.dtype).to(device).permute(1, 0, 2)

        # T, B, D
        x = self.text_pre_proj(x)
        xf_out = self.textTransEncoder(x)
        xf_out = self.text_ln(xf_out)
        if self.use_text_proj:
            xf_proj = self.text_proj(xf_out[text.argmax(dim=-1), torch.arange(xf_out.shape[1])])
            # B, T, D
            xf_out = xf_out.permute(1, 0, 2)
            return xf_proj, xf_out
        else:
            xf_out = xf_out.permute(1, 0, 2)
            return xf_out

    @abstractmethod
    def get_precompute_condition(self, **kwargs):
        pass
    
    @abstractmethod
    def forward_train(self, h, src_mask, emb, **kwargs):
        pass
    
    @abstractmethod
    def forward_test(self, h, src_mask, emb, **kwargs):
        pass

    def forward(self, motion, timesteps, motion_mask=None, **kwargs):
        """
        motion: B, T, D
        """
        B, T = motion.shape[0], motion.shape[1]
        conditions = self.get_precompute_condition(device=motion.device, **kwargs)
        if len(motion_mask.shape) == 2:
            src_mask = motion_mask.clone().unsqueeze(-1)
        else:
            src_mask = motion_mask.clone()

        if self.use_text_proj:
            emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim)) + conditions['xf_proj']
        else:
            emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim))
        # B, T, latent_dim
        h = self.joint_embed(motion)
        h = h + self.sequence_embedding.unsqueeze(0)[:, :T, :]

        if self.training:
            return self.forward_train(h=h, src_mask=src_mask, emb=emb, timesteps=timesteps, **conditions)
        else:
            return self.forward_test(h=h, src_mask=src_mask, emb=emb, timesteps=timesteps, **conditions)
