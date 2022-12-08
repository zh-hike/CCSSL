"""
author: zhhike
"""

import yaml
import torch
import paddle
import numpy as  np
import random

def load_cfg(args):
    """
    加载模型参数
    """
    path = args.config_path
    with open(path, 'r') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        args.cfg = cfg

    
def load_model_from_torch(model, path):
    print('*****************    load torch weight    *******************')
    torch_state_dict = torch.load(path, map_location='cpu')
    paddle_state_dict = model.state_dict()
    for key in torch_state_dict.keys():
        if 'num_batches_tracked' in key:
            continue
        pk = key.replace('running_mean', '_mean')
        pk = pk.replace('running_var', '_variance')
        torch_weight = torch_state_dict[key].numpy()
        if 'fc' in pk and 'weight' in pk:
            torch_weight = torch_weight.T

        paddle_state_dict[pk] = paddle.to_tensor(torch_weight)

    model.load_dict(paddle_state_dict)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)