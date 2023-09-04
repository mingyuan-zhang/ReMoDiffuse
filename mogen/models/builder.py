from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry


def build_from_cfg(cfg, registry, default_args=None):
    if cfg is None:
        return None
    return MMCV_MODELS.build_func(cfg, registry, default_args)


MODELS = Registry('models', parent=MMCV_MODELS, build_func=build_from_cfg)

LOSSES = MODELS
ARCHITECTURES = MODELS
SUBMODULES = MODELS
ATTENTIONS = MODELS

def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)

def build_architecture(cfg):
    """Build framework."""
    return ARCHITECTURES.build(cfg)

def build_submodule(cfg):
    """Build submodule."""
    return SUBMODULES.build(cfg)

def build_attention(cfg):
    """Build attention."""
    return ATTENTIONS.build(cfg)
