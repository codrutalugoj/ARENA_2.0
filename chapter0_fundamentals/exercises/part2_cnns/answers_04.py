# %%
import torch as t
import torch.nn as nn

from typing import Union, Optional, Tuple

import tests
from answers_03 import IntOrPair

MAIN = __name__ == "__main__"
# %%
from answers_03 import maxpool2d

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Call the functional version of maxpool2d.'''
        return maxpool2d(x, self.kernel_size, self.stride, self.padding)

    def extra_repr(self) -> str:
        '''Add additional information to the string representation of this class.'''
        # return f"kernel_size: {self.kernel_size}, stride: {self.stride}, padding: {self.padding}"
        # fancier way of doing it:
        # join creates a single string from the elements in the list by separating them with a comma
        return ", ".join(f"{key}={getattr(self, key)}" for key in ["kernel_size", "stride", "padding"])


if MAIN:
    tests.test_maxpool2d_module(MaxPool2d)
    m = MaxPool2d(kernel_size=3, stride=2, padding=1)
    print(f"Manually verify that this is an informative repr: {m}")


# %% 
class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.maximum(t.zeros(x.shape), x)
    
if MAIN:
    tests.test_relu(ReLU)


# %%
from functools import reduce

class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        Flatten out dimensions from start_dim to end_dim, inclusive of both.
        '''
        shape = x.shape

        start_dim = self.start_dim
        end_dim = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim

        shape_beginning = shape[:start_dim]
        shape_mid = reduce(lambda x, y: x*y, shape[start_dim : end_dim + 1])
        shape_end = shape[end_dim + 1 : ]

        reshaped_size = shape_beginning + (shape_mid,) + shape_end

        return x.reshape(reshaped_size)

    def extra_rep(self) -> str:
        return ", ".join(["{key}={getattr(self, key)}" for key in ["start_dim", "end_dim"]])
    
if MAIN:
    tests.test_flatten(Flatten)


# %% 
import einops
import numpy as np

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        init_dist = t.distributions.uniform.Uniform(- 1/np.sqrt(in_features), 1/np.sqrt(in_features))

        self.weight = nn.Parameter(init_dist.sample((out_features, in_features)))
        self.bias = nn.Parameter(init_dist.sample((out_features, ))) if bias == True else None 

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (*, in_features)
        Return: shape (*, out_features)
        '''
        out = einops.einsum(x, self.weight, "... in, out in -> ... out")

        if self.bias is not None:
            out += self.bias
        
        return out  
        

    def extra_repr(self) -> str:
        return ", ".join("{key}={getattr(self, key)}" for key in ["in_features", "out_features", "bias"])

if MAIN:
    tests.test_linear_forward(Linear)
    tests.test_linear_parameters(Linear)
    tests.test_linear_no_bias(Linear)


# %%
from answers_03 import conv2d
from answers_03 import force_pair

class Conv2d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: IntOrPair, stride: IntOrPair = 1, padding: IntOrPair = 0
    ):
        '''
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.
        '''
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = force_pair(kernel_size)
        self.stride = force_pair(stride)
        self.padding = force_pair(padding)

        kh, kw = self.kernel_size

        k = 1/(in_channels * kh * kw)
        init_dist = t.distributions.uniform.Uniform(- np.sqrt(k), np.sqrt(k))
        shape_weights = (out_channels, in_channels, kh, kw)

        self.weight = nn.Parameter(init_dist.sample(shape_weights))

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Apply the functional conv2d you wrote earlier.'''
        return conv2d(x, self.weight, self.stride, self.padding)

    def extra_repr(self) -> str:
        return ", ".join(f"{key}={getattr(self, key)}" for key in ["in_channels", 
                                                           "out_channels", 
                                                           "kernel_size",
                                                           "stride",
                                                           "padding"])


if MAIN:
    tests.test_conv2d_module(Conv2d)
    conv_layer = Conv2d(32, 64, (3,3), 1, 0)
    print(conv_layer)