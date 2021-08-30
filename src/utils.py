import torch
import numpy as np


def analyze_torch(model):
  ls = [float(m.sum()) for m in 
          model.parameters()]
  return ls 

def analyze_tf(model, key='network_fn'):
  ls = [np.sum(l) for 
         layer in model.render_kwargs_train[key].layers 
         for l in layer.get_weights() ]
  return ls
  
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def img_cvt(images):
    return (255. * images).detach().cpu().numpy().clip(0, 255).astype('uint8').transpose(1, 2, 0)