# %%

import os
import sys
import numpy as np
import einops
from typing import Union, Optional, Tuple
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
import functools
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm.notebook import tqdm

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part2_cnns"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, bar
from part2_cnns.utils import display_array_as_img, display_soln_array_as_img
import part2_cnns.tests as tests

MAIN = __name__ == "__main__"

# %%

def conv1d_minimal_simple(x: Float[Tensor, "w"], weights: Float[Tensor, "kw"]) -> Float[Tensor, "ow"]:
    '''
    Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    Simplifications: batch = input channels = output channels = 1.

    x: shape (width,)
    weights: shape (kernel_width,)

    Returns: shape (output_width,)
    '''
    width = x.shape[0]
    kernel_width = weights.shape[0]

    output_width = width - kernel_width + 1

    s_w = x.stride(0)

    x_strided = x.as_strided(size=(output_width, kernel_width), stride=(s_w, s_w))

    return einops.einsum(x_strided, weights, "ow kw, kw -> ow")


if MAIN:
    tests.test_conv1d_minimal_simple(conv1d_minimal_simple)


# %%

def conv1d_minimal(x: Float[Tensor, "b ic w"], weights: Float[Tensor, "oc ic kw"]) -> Float[Tensor, "b oc ow"]:
    '''
    Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''
    
    b, ic, w = x.shape
    oc, _, kw = weights.shape

    ow = w - kw + 1

    s_b, s_ic, s_w = x.stride()
    new_stride = (s_b, s_ic, s_w, s_w)

    x_strided = x.as_strided(size=(b, ic, ow, kw), stride=new_stride)

    return einops.einsum(x_strided, weights, "b ic ow kw, oc ic kw -> b oc ow")


if MAIN:
    tests.test_conv1d_minimal(conv1d_minimal)


# %%
def conv2d_minimal(x: Float[Tensor, "b ic h w"], weights: Float[Tensor, "oc ic kh kw"]) -> Float[Tensor, "b oc oh ow"]:
    '''
    Like torch's conv2d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    
    b, ic, h, w = x.shape
    oc, ic, kh, kw = weights.shape

    oh = h - kh + 1
    ow = w - kw + 1

    s_b, s_ic, s_h, s_w = 

    new_shape = (b, ic, oh, ow, kh, kw)
    new_stride = (s_h, s_w)


if MAIN:
    tests.test_conv2d_minimal(conv2d_minimal)