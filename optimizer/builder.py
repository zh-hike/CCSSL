"""
choices optimizer
"""

from paddle.optimizer import SGD, Momentum, Adam
import copy
from .scheduler import cosine_schedule_with_warmup

def optimizer_builder(model, cfg):
    """
    set optimizer
    """
    cf = copy.deepcopy(cfg)
    name = cf.pop('name')
    scheduler_cfg = cf.pop('scheduler')
    scheduler_name = scheduler_cfg['name']
    scheduler_cfg.pop('name')
    learning_rate = cf.pop('learning_rate')
    
    no_decay = cf.pop("no_decay", ['bias', 'bn'])
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)],
            'weight_decay': cf['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    print("********************", cf)
    # opt = eval(name)(learning_rate=learning_rate, parameters=grouped_parameters, **cf)

    scheduler = eval(scheduler_name)(learning_rate=learning_rate, **scheduler_cfg)
    opt = eval(name)(learning_rate=scheduler, parameters=grouped_parameters, **cf)

    return opt, scheduler

    # return opt