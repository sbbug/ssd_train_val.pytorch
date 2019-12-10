"""
-------------------------------------------------
   File Name:    warmup_scheduler.py
   Author:       Zhonghao Huang
   Date:         2019/9/17
   Description:
-------------------------------------------------
"""

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


def adjust_learning_rate(optimizer, gamma, step,lr):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




