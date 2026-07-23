import numpy as np
import torch
import torch.nn as nn
import threading
import cholespy
import torch.linalg as LA

import torch_mesh_ops as TMO
from torch_geometric.nn import knn


class BufferContainer(nn.Module):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        main_str = super().__repr__() + '\n'
        for name, buf in self.named_buffers():
            main_str += f'    {name:20}\t{buf.shape}\t{buf.dtype}\n'
        return main_str
    
    def __iter__(self):
        for name, buf in self.named_buffers():
            yield name, buf

    def __len__(self):
        return len(list(self.named_buffers()))
    
    def keys(self):
        return [name for name, buf in self.named_buffers()]
    
    def items(self):
        return [(name, buf) for name, buf in self.named_buffers()]



class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def to_tensor(array, dtype=torch.float32):
    if "torch.tensor" not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if "scipy.sparse" in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


def load_param(param, ckpt_array):
    param.data[:] = torch.from_numpy(ckpt_array).to(param.device)

def load_param_list(param_list, ckpt_array):
    for i in range(min(len(param_list), len(ckpt_array))):
        load_param(param_list[i], ckpt_array[i])

def async_func(func):
    """Decorator to run a function asynchronously"""
    def wrapper(*args, **kwargs):
        self = args[0]
        if self.cfg.async_func:
            thread = threading.Thread(target=func, args=args, kwargs=kwargs)
            thread.start()
        else:
            func(*args, **kwargs)
    return wrapper

def is_optimizable(name, param_groups):
    for param in param_groups:
        if name.strip() in param['name']:
            return True
    return False
