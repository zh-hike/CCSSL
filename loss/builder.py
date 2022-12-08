"""
loss select
"""

from .fixmatch_ccssl_loss import FixmatchCCSSLLoss as ccssl_loss
from .fixmatch_ccssl_loss import SoftSupConLoss
from paddle.nn import CrossEntropyLoss
import copy


def loss_builder(cfg):
    """
    select loss
    Args:
        cfg: dict, should be set in *.yaml
    """
    cf = copy.deepcopy(cfg)
    name = cf.pop('name')
    cri = eval(name)(**cf)
    return cri

