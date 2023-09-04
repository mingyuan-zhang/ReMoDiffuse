from cv2 import norm
import torch
from torch import layer_norm, nn
from mmcv.runner import BaseModule
import numpy as np

from ..builder import SUBMODULES
from .position_encoding import SinusoidalPositionalEncoding, LearnedPositionalEncoding
import math


@SUBMODULES.register_module()
class ACTOREncoder(BaseModule):
    def __init__(self,
                 max_seq_len=16,
                 njoints=None,
                 nfeats=None,
                 input_feats=None,
                 latent_dim=256,
                 output_dim=256,
                 condition_dim=None,
                 num_heads=4,
                 ff_size=1024,
                 num_layers=8,
                 activation='gelu',
                 dropout=0.1,
                 use_condition=False,
                 num_class=None,
                 use_final_proj=False,
                 output_var=False,
                 pos_embedding='sinusoidal',
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.njoints = njoints
        self.nfeats = nfeats
        if input_feats is None:
            assert self.njoints is not None and self.nfeats is not None
            self.input_feats = njoints * nfeats
        else:
            self.input_feats = input_feats
        self.max_seq_len = max_seq_len
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.use_condition = use_condition
        self.num_class = num_class
        self.use_final_proj = use_final_proj
        self.output_var = output_var
        self.skelEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.use_condition:
            if num_class is None:
                self.mu_layer = build_MLP(self.condition_dim, self.latent_dim)
                if self.output_var:
                    self.sigma_layer = build_MLP(self.condition_dim, self.latent_dim)
            else:
                self.mu_layer = nn.Parameter(torch.randn(num_class, self.latent_dim))
                if self.output_var:
                    self.sigma_layer = nn.Parameter(torch.randn(num_class, self.latent_dim))
        else:
            if self.output_var:
                self.query = nn.Parameter(torch.randn(2, self.latent_dim))
            else:
                self.query = nn.Parameter(torch.randn(1, self.latent_dim))
        if pos_embedding == 'sinusoidal':
            self.pos_encoder = SinusoidalPositionalEncoding(latent_dim, dropout)
        else:
            self.pos_encoder = LearnedPositionalEncoding(latent_dim, dropout, max_len=max_seq_len + 2)
        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation)
        self.seqTransEncoder = nn.TransformerEncoder(
            seqTransEncoderLayer,
            num_layers=num_layers)

    def forward(self, motion, motion_mask=None, condition=None):
        B, T = motion.shape[:2]
        motion = motion.view(B, T, -1)
        feature = self.skelEmbedding(motion)
        if self.use_condition:
            if self.output_var:
                if self.num_class is None:
                    sigma_query = self.sigma_layer(condition).view(B, 1, -1)
                else:
                    sigma_query = self.sigma_layer[condition.long()].view(B, 1, -1)
                feature = torch.cat((sigma_query, feature), dim=1)
            if self.num_class is None:
                mu_query = self.mu_layer(condition).view(B, 1, -1)
            else:
                mu_query = self.mu_layer[condition.long()].view(B, 1, -1)
            feature = torch.cat((mu_query, feature), dim=1)  
        else:
            query = self.query.view(1, -1, self.latent_dim).repeat(B, 1, 1)
            feature = torch.cat((query, feature), dim=1)
        if self.output_var:
            motion_mask = torch.cat((torch.zeros(B, 2).to(motion.device), 1 - motion_mask), dim=1).bool()
        else:
            motion_mask = torch.cat((torch.zeros(B, 1).to(motion.device), 1 - motion_mask), dim=1).bool()
        feature = feature.permute(1, 0, 2).contiguous()
        feature = self.pos_encoder(feature)
        feature = self.seqTransEncoder(feature, src_key_padding_mask=motion_mask)
        if self.use_final_proj:
            mu = self.final_mu(feature[0])
            if self.output_var:
                sigma = self.final_sigma(feature[1])
                return mu, sigma
            return mu
        else:
            if self.output_var:
                return feature[0], feature[1]
            else:
                return feature[0]


@SUBMODULES.register_module()
class ACTORDecoder(BaseModule):

    def __init__(self,
                 max_seq_len=16,
                 njoints=None,
                 nfeats=None,
                 input_feats=None,
                 input_dim=256,
                 latent_dim=256,
                 condition_dim=None,
                 num_heads=4,
                 ff_size=1024,
                 num_layers=8,
                 activation='gelu',
                 dropout=0.1,
                 use_condition=False,
                 num_class=None,
                 pos_embedding='sinusoidal',
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if input_dim != latent_dim:
            self.linear = nn.Linear(input_dim, latent_dim)
        else:
            self.linear = nn.Identity()
        self.njoints = njoints
        self.nfeats = nfeats
        if input_feats is None:
            assert self.njoints is not None and self.nfeats is not None
            self.input_feats = njoints * nfeats
        else:
            self.input_feats = input_feats
        self.max_seq_len = max_seq_len
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.use_condition = use_condition
        self.num_class = num_class
        if self.use_condition:
            if num_class is None:
                self.condition_bias = build_MLP(condition_dim, latent_dim)
            else:
                self.condition_bias = nn.Parameter(torch.randn(num_class, latent_dim))
        if pos_embedding == 'sinusoidal':
            self.pos_encoder = SinusoidalPositionalEncoding(latent_dim, dropout)
        else:
            self.pos_encoder = LearnedPositionalEncoding(latent_dim, dropout, max_len=max_seq_len)
        seqTransDecoderLayer = nn.TransformerDecoderLayer(
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation)
        self.seqTransDecoder = nn.TransformerDecoder(
            seqTransDecoderLayer,
            num_layers=num_layers)

        self.final = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, input, motion_mask=None, condition=None):
        B = input.shape[0]
        T = self.max_seq_len
        input = self.linear(input)
        if self.use_condition:
            if self.num_class is None:
                condition = self.condition_bias(condition)
            else:
                condition = self.condition_bias[condition.long()].squeeze(1)
            input = input + condition
        query = self.pos_encoder.pe[:T, :].view(T, 1, -1).repeat(1, B, 1)
        input = input.view(1, B, -1)
        feature = self.seqTransDecoder(tgt=query, memory=input, tgt_key_padding_mask=(1 - motion_mask).bool())
        pose = self.final(feature).permute(1, 0, 2).contiguous()
        return pose
