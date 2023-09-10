import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..builder import SUBMODULES
from .diffusion_transformer import DiffusionTransformer


@SUBMODULES.register_module()
class MotionDiffuseTransformer(DiffusionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def get_precompute_condition(self,
                                 text=None,
                                 xf_proj=None,
                                 xf_out=None,
                                 device=None,
                                 clip_feat=None,
                                 **kwargs):
        if xf_proj is None or xf_out is None:
            xf_proj, xf_out = self.encode_text(text, clip_feat, device)
        return {'xf_proj': xf_proj, 'xf_out': xf_out}

    def post_process(self, motion):
        return motion

    def forward_train(self, h=None, src_mask=None, emb=None, xf_out=None, **kwargs):
        B, T = h.shape[0], h.shape[1]
        for module in self.temporal_decoder_blocks:
            h = module(x=h, xf=xf_out, emb=emb, src_mask=src_mask)
        output = self.out(h).view(B, T, -1).contiguous()
        return output
    
    def forward_test(self, h=None, src_mask=None, emb=None, xf_out=None, **kwargs):
        B, T = h.shape[0], h.shape[1]
        for module in self.temporal_decoder_blocks:
            h = module(x=h, xf=xf_out, emb=emb, src_mask=src_mask)
        output = self.out(h).view(B, T, -1).contiguous()
        return output
