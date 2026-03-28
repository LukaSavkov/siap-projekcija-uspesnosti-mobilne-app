import numpy as np

def constant(epoch, lr_0, **kwargs):
    return lr_0

def step_decay(epoch, lr_0, F=0.5, D=500, **kwargs):
    return lr_0 * (F ** (epoch // D))

LR_SCHEDULES = {
    'constant': constant,
    'step_decay': step_decay
}