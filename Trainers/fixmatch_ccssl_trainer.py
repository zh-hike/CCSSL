"""
fixmatch ccssl trainer

"""


from pyexpat import model
import paddle
from .base_trainer import BaseTrainer
from loss.builder import loss_builder
import paddle.nn.functional as F
from tqdm import tqdm
import numpy as np
from reprod_log import ReprodLogger


class Trainer(BaseTrainer):
    """
    Fixmatch CCSSL trainer set
    """
    def __init__(self, args):
        """
        init set
        Args:
            cfg: dict, all args and it should be set in *.yaml
        """

        super().__init__(args)
        self.train_labeled_dl, self.train_unlabeled_dl, self.val_dl = self.dataladers
        self.cri_x = loss_builder(args.cfg['loss']['loss_x'])
        self.cri_u = loss_builder(args.cfg['loss']['loss_u'])
        self.cri_c = loss_builder(args.cfg['loss']['loss_c'])
        self.pseudo_with_ema = False
        if args.cfg['trainer'].get('ema', False):
            self.pseudo_with_ema = args.cfg['trainer']['ema'].get('pseudo_with_ema', False)
        
        self.lambda_u = args.cfg['loss']['lambda_u']
        self.lambda_c = args.cfg['loss']['lambda_c']

    @paddle.no_grad()
    def val(self):
        """
        start val CCSSL model
        """
        self.model.eval()
        logits = paddle.empty((0, self.args.cfg['dataset']['num_classes']))
        ema_logits = paddle.empty((0, self.args.cfg['dataset']['num_classes']))
        all_targets = paddle.empty((0,))
        for batch, (inputs, targets) in enumerate(self.val_dl):
            logit, _ = self.model(inputs[0])
            logits = paddle.concat([logits, logit], axis=0)
            if self.use_ema:
                ema_logit, _ = self.ema(inputs[0])
                ema_logits = paddle.concat([ema_logits, ema_logit], axis=0)
            all_targets = paddle.concat([all_targets, targets])
        pred = F.softmax(logits, axis=-1)
        # val_metric = self.metric(pred=pred, targets=all_targets)
        if self.use_ema:
            ema_pred = F.softmax(ema_logits, axis=-1)
        else:
            ema_pred = None
        return pred, ema_pred, all_targets

    def save(self, epoch):
        data = self.args.cfg['dataset']['name']
        trainer = self.args.cfg['trainer']['name']
        ema_state_dict = None
        if self.use_ema:
            ema_state_dict = self.ema.ema.state_dict()
        save_data = {'data': data,
                     'trainer': trainer,
                     'epoch': epoch,
                     'model': self.model.state_dict(),
                     'opt': self.opt.state_dict(),
                     'scheduler': self.scheduler.state_dict(),
                     'ema': ema_state_dict}
        paddle.save(save_data, f'./checkpoint/{trainer}_{data}.pdparams')

    def load(self):
        data = self.args.cfg['dataset']['name']
        trainer = self.args.cfg['trainer']['name']
        save_data = paddle.load(f'./checkpoint/{trainer}_{data}.pdparams')
        self.model.load_dict(save_data['model'])
        self.opt.set_state_dict(save_data['opt'])
        self.scheduler.set_state_dict(save_data['scheduler'])
        epoch = save_data['epoch']
        if self.use_ema:
            self.ema.ema.load_dict(save_data['ema'])
        return epoch

    def train_from_pretrained(self):
        epoch = self.load()

        self.train(epoch)

    def train(self, start_epoch=0):
        """
        start train CCSSL model
        """
        self.metric.set_epoch(start_epoch)
        train_labeled_dl_iter = iter(self.train_labeled_dl)
        train_unlabeled_dl_iter = iter(self.train_unlabeled_dl)
        # print(f"***********  labeled num {len(train_labeled_dl_iter)},  unlabeled num {len(train_unlabeled_dl_iter)}")
        # weight_logger = ReprodLogger()
        # dataloader_logger = ReprodLogger()
        # loss_logger = ReprodLogger()
        # out_logger = ReprodLogger()
        for epoch in range(start_epoch, self.cfg['epochs']):
            self.model.train()
            # step_per_epoch = max(len(train_labeled_dl_iter), len(train_unlabeled_dl_iter))
            pseudo_logits = paddle.empty((0, self.args.cfg['dataset']['num_classes']))
            all_targets = paddle.empty((0,))
            for idx in tqdm(range(self.args.cfg['trainer']['eval_steps']), ncols=90, disable=not self.args.bar):
                try:
                    labeled_data = next(train_labeled_dl_iter)
                except:
                    train_labeled_dl_iter = iter(self.train_labeled_dl)
                    labeled_data = next(train_labeled_dl_iter)

                try:
                    unlabeled_data = next(train_unlabeled_dl_iter)
                except:
                    train_unlabeled_dl_iter = iter(self.train_unlabeled_dl)
                    unlabeled_data = next(train_unlabeled_dl_iter)
                # state_dict = self.model.state_dict()
                # for key in state_dict:
                #     weight = state_dict[key]
                #     weight_logger.add(f'{key}_iter_{epoch}_{idx}', weight.numpy())
                


                input_x, target_x = labeled_data
                input_x = input_x[0]

                input_u, target_u = unlabeled_data
                input_w, input_s1, input_s2 = input_u
                # print(input_x.shape, input_w.shape)

                # dataloader_logger.add(f'data_x_iter_{epoch}_{idx}', input_x.numpy())
                # dataloader_logger.add(f'label_x_iter_{epoch}_{idx}', target_x.numpy())
                # dataloader_logger.add(f'label_u_iter_{epoch}_{idx}', target_u.numpy())
                # dataloader_logger.add(f'data_w_iter_{epoch}_{idx}', input_w.numpy())
                # dataloader_logger.add(f'data_s1_iter_{epoch}_{idx}', input_s1.numpy())
                # dataloader_logger.add(f'data_s2_iter_{epoch}_{idx}', input_s2.numpy())

                if not self.pseudo_with_ema:
                    n_x = input_x.shape[0]
                    x = paddle.concat([input_x, input_w, input_s1, input_s2], axis=0)
                    logits, feats = self.model(x)
                    logit_x = logits[:n_x]
                    logit_w, logit_s1, logit_s2 = logits[n_x:].chunk(3)
                    feat_w, feat_s1, feat_s2 = feats[n_x:].chunk(3)

                # out_logger.add(f'logit_x_iter_{epoch}_{idx}', logit_x.detach().numpy())
                # out_logger.add(f'logit_w_iter_{epoch}_{idx}', logit_w.detach().numpy())
                # out_logger.add(f'logit_s1_iter_{epoch}_{idx}', logit_s1.detach().numpy())
                # out_logger.add(f'logit_s2_iter_{epoch}_{idx}', logit_s2.detach().numpy())
                # out_logger.add(f'feat_x_iter_{epoch}_{idx}', feats[:n_x].detach().numpy())
                # out_logger.add(f'feat_w_iter_{epoch}_{idx}', feat_w.detach().numpy())
                # out_logger.add(f'feat_s1_iter_{epoch}_{idx}', feat_s1.detach().numpy())
                # out_logger.add(f'feat_s2_iter_{epoch}_{idx}', feat_s2.detach().numpy())


                pseudo_logits = paddle.concat([pseudo_logits, logit_w], axis=0)
                all_targets = paddle.concat([all_targets, target_u])
                results = self.compute_loss(logit_x=logit_x,
                                         target_x=target_x,
                                         logit_w=logit_w,
                                         logit_s1=logit_s1,
                                         feat_s1=feat_s1,
                                         feat_s2=feat_s2,
                                         )
                loss = results['loss']
                # loss_logger.add(f'loss_iter_{epoch}_{idx}', loss.detach().numpy())
                # loss_logger.add(f'loss_x_iter_{epoch}_{idx}', results['loss_x'].detach().numpy())
                # loss_logger.add(f'loss_u_iter_{epoch}_{idx}', results['loss_u'].detach().numpy())
                # loss_logger.add(f'loss_contrast_iter_{epoch}_{idx}', results['loss_contrast'].detach().numpy())
                # loss_logger.add(f'lr_iter_{epoch}_{idx}', np.array(self.opt.get_lr()))
                # print(f"*****************    {self.opt.get_lr()}    *****************")
                # print(loss.item())
                if self.use_ema:
                    self.ema.update(self.model)
                self.opt.clear_grad()
                loss.backward()
                self.opt.step()
                self.scheduler.step()

                # if idx == 3:
                #     break

            # weight_logger.save('/workspace/zhouhai/check/data/step1/paddle_weight.npy')
            # dataloader_logger.save('/workspace/zhouhai/check/data/step1/paddle_loader.npy')
            # loss_logger.save('/workspace/zhouhai/check/data/step2/paddle_loss.npy')
            # out_logger.save('/workspace/zhouhai/check/data/step2/paddle_out.npy')
            # assert 1==0

            pseudo_p = F.softmax(pseudo_logits, axis=-1)
            val_pred, val_ema_pred, val_targets = self.val()
            metric = self.metric(pred=val_pred,
                                 ema_pred=val_ema_pred,
                                 targets=val_targets,
                                 pseudo_pred=pseudo_p,
                                 pseudo_targets=all_targets,
                                 threshold=self.args.cfg['loss']['threshold'])
            if paddle.distributed.get_rank() in [-1, 0]:
                print(metric)
            if epoch % self.args.cfg['trainer']['save_epoch'] == 0:
                self.save(epoch)
                # pass
            # train_metric = self.metric(pred=pseudo_p, targets=all_targets, threshold=self.args.cfg['loss']['threshold'])
            # val_metric = self.val()
            # print(val_metric)

    def compute_loss(self, logit_x, target_x, logit_w, logit_s1, feat_s1, feat_s2):
        """
        compute all loss
        """
        
        loss_x = self.cri_x(logit_x, target_x)
        probs_u_w = F.softmax(logit_w / self.args.cfg['loss']['T'], axis=-1)
        max_probs, p_targets_u_w = probs_u_w.max(axis=-1), probs_u_w.argmax(axis=-1)
        threshold = self.args.cfg['loss']['threshold']
        mask = (max_probs > threshold).astype('float32')
        loss_u = (self.cri_u(logit_s1, p_targets_u_w) * mask).mean()
        feats = paddle.concat([feat_s1.unsqueeze(1), feat_s2.unsqueeze(1)], axis=1)
        loss_c = self.cri_c(feats, max_probs, p_targets_u_w)
        
        loss = loss_x + self.lambda_u * loss_u + self.lambda_c * loss_c
        results = {'loss': loss,
                   'loss_x': loss_x,
                   'loss_u': loss_u,
                   'loss_contrast': loss_c,
                   'mask_prob': mask.mean().item()}
        return results
        
    

        

