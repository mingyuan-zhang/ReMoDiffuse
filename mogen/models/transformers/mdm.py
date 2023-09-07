import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from ..builder import SUBMODULES


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp32"""

    def _convert_weights_to_fp32(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.float()
            if l.bias is not None:
                l.bias.data = l.bias.data.float()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.float()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.float()

    model.apply(_convert_weights_to_fp32)


@SUBMODULES.register_module()
class MDMTransformer(nn.Module):
    def __init__(self,
                 input_feats=263,
                 latent_dim=256,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=4,
                 dropout=0.1,
                 activation="gelu",
                 clip_dim=512,
                 clip_version=None,
                 guide_scale=1.0,
                 cond_mask_prob=0.1,
                 use_official_ckpt=False,
                 **kwargs):
        super().__init__()

        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.clip_dim = clip_dim
        self.input_feats = input_feats
        self.guide_scale = guide_scale
        self.use_official_ckpt = use_official_ckpt

        self.cond_mask_prob = cond_mask_prob
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        

        self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
        self.clip_version = clip_version
        self.clip_model = self.load_and_freeze_clip(clip_version)

        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)


    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model


    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
        return self.clip_model.encode_text(texts).float()
    
    def get_precompute_condition(self, text, device=None, **kwargs):
        if not self.training and device == torch.device('cpu'):
            convert_weights(self.clip_model)
        text_feat = self.encode_text(text)
        return {'text_feat': text_feat}

    def post_process(self, motion):
        assert len(motion.shape) == 3
        if self.use_official_ckpt:
            motion[:, :, :4] = motion[:, :, :4] * 25
        return motion

    def forward(self, motion, timesteps, text_feat=None, **kwargs):
        """
        motion: B, T, D
        timesteps: [batch_size] (int)
        """
        B, T, D = motion.shape
        device = motion.device
        if text_feat is None:
            enc_text = get_precompute_condition(**kwargs)['text_feat']
        else:
            enc_text = text_feat
        if self.training:
            # T, B, D
            motion = self.poseEmbedding(motion).permute(1, 0, 2)
            
            emb = self.embed_timestep(timesteps)  # [1, bs, d]
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=False))
           
            xseq = self.sequence_pos_encoder(torch.cat((emb, motion), axis=0))
            output = self.seqTransEncoder(xseq)[1:]

            # B, T, D
            output = self.poseFinal(output).permute(1, 0, 2)
            return output
        else:          
            # T, B, D
            motion = self.poseEmbedding(motion).permute(1, 0, 2)
            
            emb = self.embed_timestep(timesteps)  # [1, bs, d]
            emb_uncond = emb + self.embed_text(self.mask_cond(enc_text, force_mask=True))
            emb_text = emb + self.embed_text(self.mask_cond(enc_text, force_mask=False))

            xseq = self.sequence_pos_encoder(torch.cat((emb_uncond, motion), axis=0))
            xseq_text = self.sequence_pos_encoder(torch.cat((emb_text, motion), axis=0))
            output = self.seqTransEncoder(xseq)[1:]
            output_text = self.seqTransEncoder(xseq_text)[1:]
            # B, T, D
            output = self.poseFinal(output).permute(1, 0, 2)
            output_text = self.poseFinal(output_text).permute(1, 0, 2)
            scale = self.guide_scale
            output = output + scale * (output_text - output)
            return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)
