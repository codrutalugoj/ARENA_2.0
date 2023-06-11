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

    s_b, s_ic, s_h, s_w = x.stride()

    new_shape = (b, ic, oh, ow, kh, kw)
    new_stride = (s_b, s_ic, s_h, s_w, s_h, s_w)

    x_strided = x.as_strided(size=new_shape, stride=new_stride)

    return einops.einsum(x_strided, weights, "b ic oh ow kh kw, oc ic kh kw -> b oc oh ow")

if MAIN:
    tests.test_conv2d_minimal(conv2d_minimal)

# %%
def pad1d(x: t.Tensor, left: int, right: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, width), dtype float32

    Return: shape (batch, in_channels, left + right + width)
    '''
    b, ic, w = x.shape

    x_padded = x.new_full((b, ic, left+right+w), fill_value=pad_value)
    x_padded[..., left:left+w] = x

    return x_padded

if MAIN:
    tests.test_pad1d(pad1d)
    tests.test_pad1d_multi_channel(pad1d)

# %%
def pad2d(x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    '''
    b, ic, h, w = x.shape
    x_padded = x.new_full((b, ic, top + h + bottom, left + w + right), fill_value=pad_value)

    x_padded[..., top: top+h, left: left+w] = x

    return x_padded

if MAIN:
    tests.test_pad2d(pad2d)
    tests.test_pad2d_multi_channel(pad2d)


# %% 
def conv1d(
    x: Float[Tensor, "b ic w"], 
    weights: Float[Tensor, "oc ic kw"], 
    stride: int = 1, 
    padding: int = 0
) -> Float[Tensor, "b oc ow"]:
    '''
    Like torch's conv1d using bias=False.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''
    b, ic, w = x.shape
    oc, ic, kw = weights.shape
    
    x = pad1d(x, padding, padding, 0)
    ow = (w + 2 * padding - kw)//stride + 1

    s_b, s_ic, s_w = x.stride()

    new_size = (b, ic, ow, kw)
    new_stride = (s_b, s_ic, s_w * stride, s_w)

    x_extended = x.as_strided(size=new_size, stride=new_stride)

    out = einops.einsum(x_extended, weights, "b ic ow kw, oc ic kw -> b oc ow")
    
    return out

if MAIN:
    tests.test_conv1d(conv1d)

# %% 
# Helper
IntOrPair = Union[int, Tuple[int, int]]
Pair = Tuple[int, int]

def force_pair(v: IntOrPair) -> Pair:
    '''Convert v to a pair of int, if it isn't already.'''
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)

# Examples of how this function can be used:


if MAIN:
    for v in [(1, 2), 2, (1, 2, 3)]:
        try:
            print(f"{v!r:9} -> {force_pair(v)!r}")
        except ValueError:
            print(f"{v!r:9} -> ValueError")

# %%
def conv2d(
    x: Float[Tensor, "b ic h w"], 
    weights: Float[Tensor, "oc ic kh kw"], 
    stride: IntOrPair = 1, 
    padding: IntOrPair = 0
) -> Float[Tensor, "b oc oh ow"]:
    '''
    Like torch's conv2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    b, ic, h, w = x.shape
    oc, ic, kh, kw = weights.shape

    pad_h, pad_w = force_pair(padding)
    stride_h, stride_w = force_pair(stride)

    x = pad2d(x, left=pad_w, right=pad_w, bottom=pad_h, top=pad_h, pad_value=0)

    s_b, s_ic, s_h, s_w = x.stride()

    oh = (h + 2*pad_h - kh)//stride_h + 1
    ow = (w + 2*pad_w - kw)//stride_w + 1

    new_size = (b, ic, oh, ow, kh, kw)
    new_stride = (s_b, s_ic, s_h * stride_h, s_w * stride_w, s_h, s_w)

    x_strided = x.as_strided(size=new_size, stride=new_stride)
    out = einops.einsum(x_strided, weights, "b ic oh ow kh kw, oc ic kh kw -> b oc oh ow")

    return out


if MAIN:
    tests.test_conv2d(conv2d)


# %%
# BUGGY
def maxpool2d(
    x: Float[Tensor, "b ic h w"], 
    kernel_size: IntOrPair, 
    stride: Optional[IntOrPair] = None, 
    padding: IntOrPair = 0
) -> Float[Tensor, "b ic oh ow"]:
    '''
    Like PyTorch's maxpool2d.

    x: shape (batch, channels, height, width)
    stride: if None, should be equal to the kernel size

    Return: (batch, channels, output_height, output_width)
    '''
    
    if stride is None:
        stride = kernel_size

    stride_h, stride_w = force_pair(stride)
    padding_h, padding_w = force_pair(padding)
    kh, kw = force_pair(kernel_size)

    b, ic, h, w = x.shape

    oh = (h + 2*padding_h - kh)//stride_h + 1
    ow = (w + 2*padding_w - kw)//stride_w + 1

    x_padded = pad2d(x, left=padding_w, top=padding_h, right=padding_w, bottom=padding_h, pad_value=-t.inf)
    s_b, s_ic, s_h, s_w = x.stride()

    new_size = (b, ic, oh, ow, kh, kw)
    new_stride = (s_b, s_ic, s_h*stride_h, s_w*stride_w, s_h, s_w)

    x_strided = x_padded.as_strided(size=new_size, stride=new_stride)

    out = t.amax(x_strided, dim=(-1, -2))

    return out

if MAIN:
    tests.test_maxpool2d(maxpool2d)