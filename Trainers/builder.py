"""

choice trainer
"""

from .fixmatch_ccssl_trainer import Trainer as FixmatchCCSSL_Trainer
import paddle.distributed as dist

def trainer_builder(args):
    """
    choice trainer
    """
    
    # print("********************   init distribute %d **********************" % dist.get_rank())

    name = args.cfg['trainer']['name']
    trainer = eval(name)(args)
    

    return trainer
