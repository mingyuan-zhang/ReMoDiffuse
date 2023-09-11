from cv2 import norm
import torch
import torch.nn.functional as F
from torch import layer_norm, nn
import numpy as np
import clip
import random
import math

from ..builder import SUBMODULES, build_attention
from .diffusion_transformer import DiffusionTransformer


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


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
                

class FFN(nn.Module):

    def __init__(self, latent_dim, ffn_dim, dropout):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        y = x + y
        return y


class EncoderLayer(nn.Module):

    def __init__(self,
                 sa_block_cfg=None,
                 ca_block_cfg=None,
                 ffn_cfg=None):
        super().__init__()
        self.sa_block = build_attention(sa_block_cfg)
        self.ffn = FFN(**ffn_cfg)

    def forward(self, **kwargs):
        if self.sa_block is not None:
            x = self.sa_block(**kwargs)
            kwargs.update({'x': x})
        if self.ffn is not None:
            x = self.ffn(**kwargs)
        return x


class RetrievalDatabase(nn.Module):

    def __init__(self,
                 num_retrieval=None,
                 topk=None,
                 retrieval_file=None,
                 latent_dim=512,
                 output_dim=512,
                 num_layers=2,
                 num_motion_layers=4,
                 kinematic_coef=0.1,
                 max_seq_len=196,
                 num_heads=8,
                 ff_size=1024,
                 stride=4,
                 sa_block_cfg=None,
                 ffn_cfg=None,
                 dropout=0):
        super().__init__()
        self.num_retrieval = num_retrieval
        self.topk = topk
        self.latent_dim = latent_dim
        self.stride = stride
        self.kinematic_coef = kinematic_coef
        self.num_layers = num_layers
        self.num_motion_layers = num_motion_layers
        self.max_seq_len = max_seq_len
        data = np.load(retrieval_file)
        self.text_features = torch.Tensor(data['text_features'])
        self.captions = data['captions']
        self.motions = data['motions']
        self.m_lengths = data['m_lengths']
        self.clip_seq_features = data['clip_seq_features']
        self.train_indexes = data.get('train_indexes', None)
        self.test_indexes = data.get('test_indexes', None)

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.motion_proj = nn.Linear(self.motions.shape[-1], self.latent_dim)
        self.motion_pos_embedding = nn.Parameter(torch.randn(max_seq_len, self.latent_dim))
        self.motion_encoder_blocks = nn.ModuleList()
        for i in range(num_motion_layers):
            self.motion_encoder_blocks.append(
                EncoderLayer(
                    sa_block_cfg=sa_block_cfg,
                    ffn_cfg=ffn_cfg
                )
            )
        TransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation="gelu")
        self.text_encoder = nn.TransformerEncoder(
            TransEncoderLayer,
            num_layers=num_layers)
        self.results = {}

    def extract_text_feature(self, text, clip_model, device):
        text = clip.tokenize([text], truncate=True).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text)
        return text_features
    
    def encode_text(self, text, device):
        with torch.no_grad():
            text = clip.tokenize(text, truncate=True).to(device)
            x = self.clip.token_embedding(text).type(self.clip.dtype)  # [batch_size, n_ctx, d_model]

            x = x + self.clip.positional_embedding.type(self.clip.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.clip.transformer(x)
            x = self.clip.ln_final(x).type(self.clip.dtype)

        # B, T, D
        xf_out = x.permute(1, 0, 2)
        return xf_out

    def retrieve(self, caption, length, clip_model, device, idx=None):
        if self.training and self.train_indexes is not None and idx is not None:
            idx = idx.item()
            indexes = self.train_indexes[idx]
            data = []
            cnt = 0
            for retr_idx in indexes:
                if retr_idx != idx:
                    data.append(retr_idx)
                    cnt += 1
                    if cnt == self.topk:
                        break
            random.shuffle(data)
            return data[:self.num_retrieval]
        
        elif not self.training and self.test_indexes is not None and idx is not None:
            idx = idx.item()
            indexes = self.test_indexes[idx]
            data = []
            cnt = 0
            for retr_idx in indexes:
                data.append(retr_idx)
                cnt += 1
                if cnt == self.topk:
                    break
            # random.shuffle(data)
            return data[:self.num_retrieval]
        else:
            value = hash(caption)
            if value in self.results:
                return self.results[value]
            text_feature = self.extract_text_feature(caption, clip_model, device)
            
            rel_length = torch.LongTensor(self.m_lengths).to(device)
            rel_length = torch.abs(rel_length - length) / torch.clamp(rel_length, min=length)
            semantic_score = F.cosine_similarity(self.text_features.to(device), text_feature)
            kinematic_score = torch.exp(-rel_length * self.kinematic_coef)
            score = semantic_score * kinematic_score
            indexes = torch.argsort(score, descending=True)
            data = []
            cnt = 0
            for idx in indexes:
                caption, motion, m_length = self.captions[idx], self.motions[idx], self.m_lengths[idx]
                if not self.training or m_length != length:
                    cnt += 1
                    data.append(idx.item())
                    if cnt == self.num_retrieval:
                        self.results[value] = data
                        return data
        assert False

    def generate_src_mask(self, T, length):
        B = len(length)
        src_mask = torch.ones(B, T)
        for i in range(B):
            for j in range(length[i], T):
                src_mask[i, j] = 0
        return src_mask

    def forward(self, captions, lengths, clip_model, device, idx=None):
        B = len(captions)
        all_indexes = []
        for b_ix in range(B):
            length = int(lengths[b_ix])
            if idx is None:
                batch_indexes = self.retrieve(captions[b_ix], length, clip_model, device)
            else:
                batch_indexes = self.retrieve(captions[b_ix], length, clip_model, device, idx[b_ix])
            all_indexes.extend(batch_indexes)
        all_indexes = np.array(all_indexes)
        N = all_indexes.shape[0]
        all_motions = torch.Tensor(self.motions[all_indexes]).to(device)
        all_m_lengths = torch.Tensor(self.m_lengths[all_indexes]).long()
        all_captions = self.captions[all_indexes].tolist()
            
        T = all_motions.shape[1]
        src_mask = self.generate_src_mask(T, all_m_lengths).to(device)
        raw_src_mask = src_mask.clone()
        re_motion = self.motion_proj(all_motions) + self.motion_pos_embedding.unsqueeze(0)
        for module in self.motion_encoder_blocks:
            re_motion = module(x=re_motion, src_mask=src_mask.unsqueeze(-1))
        re_motion = re_motion.view(B, self.num_retrieval, T, -1).contiguous()
        # stride
        re_motion = re_motion[:, :, ::self.stride, :].contiguous()
        
        src_mask = src_mask[:, ::self.stride].contiguous()
        src_mask = src_mask.view(B, self.num_retrieval, -1).contiguous()

        T = 77
        all_text_seq_features = torch.Tensor(self.clip_seq_features[all_indexes]).to(device)
        all_text_seq_features = all_text_seq_features.permute(1, 0, 2)
        re_text = self.text_encoder(all_text_seq_features)
        re_text = re_text.permute(1, 0, 2).view(B, self.num_retrieval, T, -1).contiguous()
        re_text = re_text[:, :, -1:, :].contiguous()
        
        # T = re_motion.shape[2]
        # re_feat = re_feat.view(B, self.num_retrieval * T, -1).contiguous()
        re_dict = dict(
            re_text=re_text,
            re_motion=re_motion,
            re_mask=src_mask,
            raw_motion=all_motions,
            raw_motion_length=all_m_lengths,
            raw_motion_mask=raw_src_mask)
        return re_dict


@SUBMODULES.register_module()
class ReMoDiffuseTransformer(DiffusionTransformer):
    def __init__(self,
                 retrieval_cfg=None,
                 scale_func_cfg=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.database = RetrievalDatabase(**retrieval_cfg)
        self.scale_func_cfg = scale_func_cfg
        
    def scale_func(self, timestep):
        coarse_scale = self.scale_func_cfg['coarse_scale']
        w = (1 - (1000 - timestep) / 1000) * coarse_scale + 1
        if timestep > 100:
            if random.randint(0, 1) == 0:
                output = {
                    'both_coef': w,
                    'text_coef': 0,
                    'retr_coef': 1 - w,
                    'none_coef': 0
                }
            else:
                output = {
                    'both_coef': 0,
                    'text_coef': w,
                    'retr_coef': 0,
                    'none_coef': 1 - w
                }
        else:
            both_coef = self.scale_func_cfg['both_coef']
            text_coef = self.scale_func_cfg['text_coef']
            retr_coef = self.scale_func_cfg['retr_coef']
            none_coef = 1 - both_coef - text_coef - retr_coef
            output = {
                'both_coef': both_coef,
                'text_coef': text_coef,
                'retr_coef': retr_coef,
                'none_coef': none_coef
            }
        return output
            
    def get_precompute_condition(self, 
                                 text=None,
                                 motion_length=None,
                                 xf_out=None,
                                 re_dict=None,
                                 device=None,
                                 sample_idx=None,
                                 clip_feat=None,
                                 **kwargs):
        if xf_out is None:
            xf_out = self.encode_text(text, clip_feat, device)
        output = {'xf_out': xf_out}
        if re_dict is None:
            re_dict = self.database(text, motion_length, self.clip, device, idx=sample_idx)
        output['re_dict'] = re_dict
        return output

    def post_process(self, motion):
        return motion

    def forward_train(self, h=None, src_mask=None, emb=None, xf_out=None, re_dict=None, **kwargs):
        B, T = h.shape[0], h.shape[1]
        cond_type = torch.randint(0, 100, size=(B, 1, 1)).to(h.device)
        for module in self.temporal_decoder_blocks:
            h = module(x=h, xf=xf_out, emb=emb, src_mask=src_mask, cond_type=cond_type, re_dict=re_dict)

        output = self.out(h).view(B, T, -1).contiguous()
        return output
    
    def forward_test(self, h=None, src_mask=None, emb=None, xf_out=None, re_dict=None, timesteps=None, **kwargs):
        B, T = h.shape[0], h.shape[1]
        both_cond_type = torch.zeros(B, 1, 1).to(h.device) + 99
        text_cond_type = torch.zeros(B, 1, 1).to(h.device) + 1
        retr_cond_type = torch.zeros(B, 1, 1).to(h.device) + 10
        none_cond_type = torch.zeros(B, 1, 1).to(h.device)
        
        all_cond_type = torch.cat((
            both_cond_type, text_cond_type, retr_cond_type, none_cond_type
        ), dim=0)
        h = h.repeat(4, 1, 1)
        xf_out = xf_out.repeat(4, 1, 1)
        emb = emb.repeat(4, 1)
        src_mask = src_mask.repeat(4, 1, 1)
        if re_dict['re_motion'].shape[0] != h.shape[0]:
            re_dict['re_motion'] = re_dict['re_motion'].repeat(4, 1, 1, 1)
            re_dict['re_text'] = re_dict['re_text'].repeat(4, 1, 1, 1)
            re_dict['re_mask'] = re_dict['re_mask'].repeat(4, 1, 1)
        for module in self.temporal_decoder_blocks:
            h = module(x=h, xf=xf_out, emb=emb, src_mask=src_mask, cond_type=all_cond_type, re_dict=re_dict)
        out = self.out(h).view(4 * B, T, -1).contiguous()
        out_both = out[:B].contiguous()
        out_text = out[B: 2 * B].contiguous()
        out_retr = out[2 * B: 3 * B].contiguous()
        out_none = out[3 * B:].contiguous()
        
        coef_cfg = self.scale_func(int(timesteps[0]))
        both_coef = coef_cfg['both_coef']
        text_coef = coef_cfg['text_coef']
        retr_coef = coef_cfg['retr_coef']
        none_coef = coef_cfg['none_coef']
        output = out_both * both_coef + out_text * text_coef + out_retr * retr_coef + out_none * none_coef
        return output
