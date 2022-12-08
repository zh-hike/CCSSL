"""
load dataset
"""

from .cifar import CIFAR10SSL, x_u_split
from paddle.io import DataLoader, RandomSampler, DistributedBatchSampler
from paddle import distributed

def dataset_builder(cfg, dist=False):
    """
    get dataloader
    Args:
        cfg: dict which should be set in *.yaml
    """
    # get_sampler = lambda x: DistributedBatchSampler if dist else RandomSampler
    name = cfg['name']
    world_size = distributed.get_world_size()
    unlabeled_train_data = eval(name)(root=cfg['data']['root'], 
                                index=None, 
                                transforms=cfg['data']['train_transform']['unlabeld_transform'],
                                mode='train')
    labeled_index, unlabeled_index = x_u_split(cfg, unlabeled_train_data.y)
    # print(f"**************   {len(labeled_index)},  {len(unlabeled_index)}  ****************")
    labeled_train_data = eval(name)(root=cfg['data']['root'],
                                    index=labeled_index,
                                    transforms=cfg['data']['train_transform']['labeld_transform'],
                                    mode='train')

    val_data = eval(name)(root=cfg['data']['root'],
                          index=None,
                          transforms=cfg['data']['val_transform'],
                          mode='test'
                        )
    
    labeled_train_sampler = get_sampler(labeled_train_data, cfg, dist)
    unlabeled_train_sampler = get_sampler(unlabeled_train_data, cfg, dist)
    val_sampler = get_sampler(val_data, cfg, dist)

    labeled_train_dl = DataLoader(labeled_train_data, 
                                #   shuffle=False, 
                                  batch_sampler=labeled_train_sampler,
                                #   batch_size=cfg['batch_size'], 
                                  num_workers=cfg['num_workers'],
                                #   drop_last=True,
                                  )
    unlabeled_train_dl = DataLoader(unlabeled_train_data,
                                    # shuffle=False,
                                    batch_sampler=unlabeled_train_sampler,
                                    # batch_size=cfg['batch_size'] * cfg['mu'],
                                    num_workers=cfg['num_workers'],
                                    # drop_last=True,
                                    )
    val_dl = DataLoader(val_data,
                        # shuffle=False,
                        batch_sampler=val_sampler,
                        # batch_size=cfg['batch_size'],
                        num_workers=cfg['num_workers'])

    return labeled_train_dl, unlabeled_train_dl, val_dl



def get_sampler(data, cfg, dist):
    if dist:
        sampler = DistributedBatchSampler(data, batch_size=cfg['batch_size'], shuffle=True, drop_last=True)
    else:
        sampler = RandomSampler(data)

    return sampler