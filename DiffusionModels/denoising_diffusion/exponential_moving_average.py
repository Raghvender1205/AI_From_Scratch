# Exponential Moving Average (EMA)
# https://medium.datadriveninvestor.com/exponentially-weighted-average-for-deep-neural-networks-39873b8230e9
"""
EMA is a Moving average which applies more weight to the recent data points 
rather than those which happened in the past. It essentially weighs the number of 
observations and use their average.
"""
import torch
from torch import nn
import copy

def exists(val):
    return val is not None

def is_float_dtype(dtype):
    return any([dtype == float_dtype for float_dtype in 
                (torch.float64, torch.float32, torch.float16, torch.bfloat16)])

def clamp(value, min_value=None, max_value=None):
    assert exists(min_value) or exists(max_value)
    if exists(min_value):
        value = max(value, min_value)
    if exists(max_value):
        value = min(value, max_value)

    return value

class EMA(nn.Module):
    """
    Implementation of Exponential Moving Average Shadowing of your model

    It uses an inner decay scheule to manage longer term training runs by adjusting power, you can
    control how fast EMA will ramp up to your specified beta.

    If gamma = 1 and power = 1, implements a simple average. gamma = 1 and power = 2/3 are good values
    for models you plan to train for a million or more steps reaches decay factor 0.999 at 31.6K steps, 
    0.9999 at 1M steps), gamma=1, power=3/4 for models you plan to train for less 
    (reaches decay factor 0.999 at 10K steps, 0.9999 at 215.4k steps).
    
    Args:
        - inv_gamma (Float): Inverse multiplicative factor of EMA.
        - power (Float): Exponential factor of EMA warmup.
        - min_value (Float): Minimum EMA decay rate
    """
    def __init__(
        self,
        model,
        # If your model has lazylinears or other types of non-deepcopyable modules, you can pass in your own ema model
        ema_model: nn.Module = None, 
        beta = 0.9999,
        update_after_step = 100,
        update_every = 10,
        inv_gamma = 1.0,
        power = 2 / 3,
        min_value = 0.0,
        param_or_buffer_names_no_ema = set(),
        ignore_names = set(),
    ):
        super().__init__()
        self.beta = beta
        self.online_model = model
        self.ema_model = ema_model

        if not exists(self.ema_model):
            try:
                self.ema_model = copy.deepcopy(model)
            except:
                print('Your model is not copyable. Please make sure you are not using LazyLinear')
                exit()

        self.ema_model.requires_grad_(False)
        self.update_every = update_every
        self.update_after_step = update_after_step

        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value

        assert isinstance(param_or_buffer_names_no_ema, (set, list))
        self.param_or_buffer_names_no_ema = param_or_buffer_names_no_ema # parameter or buffer

        self.ignore_names = ignore_names
        self.register_buffer('initted', torch.Tensor([False]))
        self.register_buffer('step', torch.Tensor([0]))

    def restore_ema_model_device(self):
        device = self.initted.device
        self.ema_model.to(device)
    
    def copy_params_from_model_to_ema(self):
        for ma_params, curr_params in zip(list(self.ema_model.parameters()), list(self.online_model.parameters())):
            if not is_float_dtype(curr_params.dtype):
                continue
                
            ma_params.data.copy_(curr_params.data)
        
        for ma_buffers, curr_buffers in zip(list(self.ema_model.buffers()), list(self.online_model.buffers())):
            if not is_float_dtype(curr_buffers.dtype):
                continue
                
            ma_buffers.data.copy_(curr_buffers.data)

    def get_curr_decay(self):
        epoch = clamp(self.step.time() - self.update_after_step - 1, min_value=0.)
        value = 1 - (1 + epoch / self.inv_gamma) ** - self.power

        if epoch <= 0:
            epoch = 0
        
        return clamp(value, min_value=self.min_value, max_value=self.beta)

    def update(self):
        step = self.step.item()
        self.step += 1

        if (step % self.update_every) != 0:
            return
        
        if step <= self.update_after_step:
            self.copy_params_from_model_to_ema()
            return

        if not self.initted.item():
            self.copy_params_from_model_to_ema()
            self.initted.data.copy_(torch.Tensor([True]))
        
        self.update_moving_average(self.ema_model, self.online_model)
    
    # Update Moving Average
    @torch.no_grad()
    def update_moving_average(self, ema_model: nn.Module, curr_model: nn.Module):
        curr_decay = self.get_curr_decay()

        for(name, curr_params), (_, ma_params) in zip(list(curr_model.named_parameters()), list(ema_model.named_parameters())):
            if name in self.ignore_names:
                continue
        
            if not is_float_dtype(curr_params.dtype):
                continue
                
            if name in self.param_or_buffer_names_no_ema:
                ma_params.data.copy_(curr_params.data)
                continue
            
            diff = ma_params.data - curr_params.data
            diff.mul_(1.0 - curr_decay)
            ma_params.sub_(diff)
        
        for (name, curr_buffer), (_, ma_buffer) in zip(list(curr_model.named_buffers()), list(ema_model.named_buffers())):
            if name in self.ignore_names:
                continue
            
            if not is_float_dtype(curr_buffer.dtype):
                continue

            if name in self.param_or_buffer_names_no_ema:
                ma_buffer.data.copy_(curr_buffer.data)
                continue
                
            diff = ma_buffer - curr_buffer
            diff.mul_(1.0 - curr_decay)
            ma_buffer.sub_(diff)
        
    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)