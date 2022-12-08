"""
choices model
"""

from .wideresnet import WideResNet as wideresnet
import copy
import os
from utils.util import load_model_from_torch

def model_builder(cfg):
    """
    choice model
    Args:
        cfg: dict which should be set in *.yaml
    """

    
    cf = copy.deepcopy(cfg)
    name = cf.pop('name')
    model = eval(name)(**cf)
    # load_model_from_torch(model, '/workspace/zhouhai/check/data/model_state/pytorch.pth')
    return model