"""
cal all metric
"""
from .acc import top1_acc, top1_pseudo_acc
import time
from paddle import distributed as dist
import paddle


class MetricBuilder:
    """
    build a mul-metrics class
    """
    def __init__(self, cfg, use_dist=False, epoch=1):
        self.epoch = 1
        self.cfg = cfg
        self.start = time.time()
        self.best_top1_acc = 0
        self.use_dist = use_dist

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __call__(self, **kargs):
        metric = {"epoch": self.epoch,
                  "cost(s)": round(time.time() - self.start, 2)}
        self.start = time.time()
        self.epoch += 1
        for name in self.cfg:
            v = eval(name)(**kargs)
            if self.use_dist:
                new_v = {}
                for key in v:
                    vs = []
                    value = paddle.to_tensor(v[key])
                    dist.all_gather(vs, value)
                    new_v[key] = paddle.concat(vs).mean().item()
                v = new_v
                
            metric[name] = v
        self.best_top1_acc = max(self.best_top1_acc, metric['top1_acc']['acc'], metric['top1_acc']['ema_acc'])
        metric['best_top1_acc'] = self.best_top1_acc
        return metric

