import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.stylization_block import StylizationBlock
from ..builder import ATTENTIONS


@ATTENTIONS.register_module()
class BaseMixedAttention(nn.Module):

    def __init__(self, latent_dim,
                       text_latent_dim,
                       num_heads,
                       dropout,
                       time_embed_dim):
        super().__init__()
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key_text = nn.Linear(text_latent_dim, latent_dim)
        self.value_text = nn.Linear(text_latent_dim, latent_dim)
        self.key_motion = nn.Linear(latent_dim, latent_dim)
        self.value_motion = nn.Linear(latent_dim, latent_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
    
    def forward(self, x, xf, emb, src_mask, cond_type, **kwargs):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = x.shape
        N = xf.shape[1] + x.shape[1]
        H = self.num_heads
        # B, T, D
        query = self.query(self.norm(x)).view(B, T, H, -1)
        # B, N, D
        text_cond_type = ((cond_type % 10) > 0).float().view(B, 1, 1).repeat(1, xf.shape[1], 1)
        key = torch.cat((
            self.key_text(self.text_norm(xf)),
            self.key_motion(self.norm(x))
        ), dim=1).view(B, N, H, -1)
        
        attention = torch.einsum('bnhl,bmhl->bnmh', query, key)
        motion_mask = src_mask.view(B, 1, T, 1)
        text_mask = text_cond_type.view(B, 1, -1, 1)
        mask = torch.cat((text_mask, motion_mask), dim=2)
        attention = attention + (1 - mask) * -1000000
        attention = F.softmax(attention, dim=2)
        
        value = torch.cat((
            self.value_text(self.text_norm(xf)) * text_cond_type,
            self.value_motion(self.norm(x)) * src_mask,
        ), dim=1).view(B, N, H, -1)
        
        y = torch.einsum('bnmh,bmhl->bnhl', attention, value).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y
    
    
@ATTENTIONS.register_module()
class BaseSelfAttention(nn.Module):

    def __init__(self, latent_dim,
                       num_heads,
                       dropout,
                       time_embed_dim):
        super().__init__()
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
    
    def forward(self, x, emb, src_mask, **kwargs):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        H = self.num_heads
        # B, T, D
        query = self.query(self.norm(x)).view(B, T, H, -1)
        # B, N, D
        key = self.key(self.norm(x)).view(B, T, H, -1)
        
        attention = torch.einsum('bnhl,bmhl->bnmh', query, key)
        mask = src_mask.view(B, 1, T, 1)
        attention = attention + (1 - mask) * -1000000
        attention = F.softmax(attention, dim=2)
        value = (self.value(self.norm(x)) * src_mask).view(B, T, H, -1)
        y = torch.einsum('bnmh,bmhl->bnhl', attention, value).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y
    

@ATTENTIONS.register_module()
class BaseCrossAttention(nn.Module):

    def __init__(self, latent_dim,
                       text_latent_dim,
                       num_heads,
                       dropout,
                       time_embed_dim):
        super().__init__()
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
    
    def forward(self, x, xf, emb, src_mask, cond_type, **kwargs):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_heads
        # B, T, D
        query = self.query(self.norm(x)).view(B, T, H, -1)
        # B, N, D
        text_cond_type = ((cond_type % 10) > 0).float().view(B, 1, 1).repeat(1, xf.shape[1], 1)
        key = self.key(self.text_norm(xf)).view(B, N, H, -1)
        attention = torch.einsum('bnhl,bmhl->bnmh', query, key)
        mask = text_cond_type.view(B, 1, -1, 1)
        attention = attention + (1 - mask) * -1000000
        attention = F.softmax(attention, dim=2)
        
        value = (self.value(self.text_norm(xf)) * text_cond_type).view(B, N, H, -1)
        y = torch.einsum('bnmh,bmhl->bnhl', attention, value).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y
