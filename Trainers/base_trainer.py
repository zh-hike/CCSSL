
"""
provide a base trainer for other method trainer.
author: zhhike
date: 2022.11.25
"""

from loss.builder import loss_builder
from dataset.builder import dataset_builder
from models.builder import model_builder
from metric.builder import MetricBuilder
from optimizer.builder import optimizer_builder
import paddle
import paddle.nn as nn
from models.ema import ModelEMA


class BaseTrainer(object):
    def __init__(self, args) -> None:
        self.args = args
        self.cfg = args.cfg['trainer']
        self.model = model_builder(args.cfg['model'])
        self.use_ema = False
        self.ema = None
        self.dist = False
        if self.cfg.get('ema', False) and self.cfg['ema']['use']:
            print("*********   use ema  ***********")
            self.use_ema = True
            self.ema = ModelEMA(self.model, self.cfg['ema']['decay'])
        if args.dist:
            self.model = paddle.DataParallel(self.model)
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.dist = True
        self.opt, self.scheduler = optimizer_builder(self.model, args.cfg['trainer']['optimizer'])
        # self.opt = optimizer_builder(self.model, args.cfg['trainer']['optimizer'])
        # self.cri = loss_builder(args.cfg['loss'])
        self.dataladers = dataset_builder(args.cfg['dataset'], self.args.dist)
        self.metric = MetricBuilder(args.cfg['metric'], use_dist=self.dist)

    def compute_loss(self, **kargs):
        pass


    def metric(self, **kargs):
        metric = self.metric(**kargs)

        return metric

