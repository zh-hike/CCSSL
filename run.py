"""
start here.
"""

import argparse

import paddle
from utils.util import load_cfg
from Trainers.builder import trainer_builder
from paddle import distributed as dist
import warnings
from utils.util import set_seed
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='fixmatch CCSSL')
    parser.add_argument('--config_path', 
                        type=str, 
                        help="path of config file", 
                        default='./config/fixmatchccssl_cifar10.yaml')

    parser.add_argument('--cfg', help="model config")
    parser.add_argument('--dist', action='store_true', help="use mul-gpu?")
    parser.add_argument('--out',
                        type=str,
                        help="path of out",
                        default='./results/')
    parser.add_argument('--bar', action='store_true', help="show bar?")


    args = parser.parse_args()
    return args

args = parse_args()
load_cfg(args)
paddle.set_device('gpu')

if __name__ == "__main__":
    
    # set_seed(0)
    if args.dist:
        dist.init_parallel_env()

    print(dist.get_world_size())
    trainer = trainer_builder(args)
    # trainer.train_from_pretrained()
    trainer.train()

