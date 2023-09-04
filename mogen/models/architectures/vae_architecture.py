import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_architecture import BaseArchitecture
from ..builder import (
    ARCHITECTURES,
    build_architecture,
    build_submodule,
    build_loss
)


@ARCHITECTURES.register_module()
class PoseVAE(BaseArchitecture):

    def __init__(self,
                 encoder=None,
                 decoder=None,
                 loss_recon=None,
                 kl_div_loss_weight=None,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.encoder = build_submodule(encoder)
        self.decoder = build_submodule(decoder)
        self.loss_recon = build_loss(loss_recon)
        self.kl_div_loss_weight = kl_div_loss_weight

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)

        eps = std.data.new(std.size()).normal_()
        latent_code = eps.mul(std).add_(mu)
        return latent_code

    def encode(self, pose):
        mu, logvar = self.encoder(pose)
        return mu

    def forward(self, **kwargs):
        motion = kwargs['motion'].float()
        B, T = motion.shape[:2]
        pose = motion.reshape(B * T, -1)
        pose = pose[:, :-4]

        mu, logvar = self.encoder(pose)
        z = self.reparameterize(mu, logvar)
        pred = self.decoder(z)

        loss = dict()
        recon_loss = self.loss_recon(pred, pose, reduction_override='none')
        loss['recon_loss'] = recon_loss
        if self.kl_div_loss_weight is not None:
            loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss['kl_div_loss'] = (loss_kl * self.kl_div_loss_weight)

        return loss
    

@ARCHITECTURES.register_module()
class MotionVAE(BaseArchitecture):

    def __init__(self,
                 encoder=None,
                 decoder=None,
                 loss_recon=None,
                 kl_div_loss_weight=None,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.encoder = build_submodule(encoder)
        self.decoder = build_submodule(decoder)
        self.loss_recon = build_loss(loss_recon)
        self.kl_div_loss_weight = kl_div_loss_weight

    def sample(self, std=1, latent_code=None):
        if latent_code is not None:
            z = latent_code
        else:
            z = torch.randn(1, 7, self.decoder.latent_dim).cuda() * std
        output = self.decoder(z)
        if self.use_normalization:
            output = output * self.motion_std
            output = output + self.motion_mean
        return output

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)

        eps = std.data.new(std.size()).normal_()
        latent_code = eps.mul(std).add_(mu)
        return latent_code

    def encode(self, motion, motion_mask):
        mu, logvar = self.encoder(motion, motion_mask)
        return self.reparameterize(mu, logvar)
    
    def decode(self, z, motion_mask):
        return self.decoder(z, motion_mask)

    def forward(self, **kwargs):
        motion, motion_mask = kwargs['motion'].float(), kwargs['motion_mask']
        B, T = motion.shape[:2]

        mu, logvar = self.encoder(motion, motion_mask)
        z = self.reparameterize(mu, logvar)
        pred = self.decoder(z, motion_mask)
        
        loss = dict()
        recon_loss = self.loss_recon(pred, motion, reduction_override='none')
        recon_loss = (recon_loss.mean(dim=-1) * motion_mask).sum() / motion_mask.sum()
        loss['recon_loss'] = recon_loss
        if self.kl_div_loss_weight is not None:
            loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss['kl_div_loss'] = (loss_kl * self.kl_div_loss_weight)

        return loss