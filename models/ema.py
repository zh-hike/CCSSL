"""
The code is set ema model
"""

import paddle
import copy


""" The Code is under Tencent Youtu Public Rule
"""
import copy
import paddle


# Exponential Moving Average
# used in teacher-student based SSL althorithm, e.g., FixMatch

# args:
#     decay(float): weight dacay

"""
Exponential Moving Average
used in teacher-student based SSL althorithm, e.g., FixMatch

args:
    decay(float): weight dacay
"""

class ModelEMA(object):
    def __init__(self, model, decay):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'module')
        # Fix EMA. https://github.com/valencebond/FixMatch_pytorch thank you!
        self.param_key = [k for k, _ in self.ema.named_parameters()]
        self.buffer_key = [k for k, _ in self.ema.named_buffers()]
        for param in self.ema.parameters():
            param.stop_gradident=True
            
    def update(self, model):
        msd = model.state_dict()
        esd = self.ema.state_dict()
        with paddle.no_grad():
            for key in self.param_key:
                model_v = msd[key].detach()
                ema_v = esd[key]
                esd[key] = self.decay * ema_v + (1 - self.decay) * model_v
                
            for key in self.buffer_key:
                esd[key] = msd[key].detach()
                
        self.ema.load_dict(esd)

    @paddle.no_grad()
    def __call__(self, data):
        return self.ema(data)

