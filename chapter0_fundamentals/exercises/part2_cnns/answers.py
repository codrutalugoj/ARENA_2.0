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

arr = np.load(section_dir / "numbers.npy")
print(type(arr), arr.shape)
# %%
if MAIN:
    display_array_as_img(arr[0])

# %%

arr1 = einops.rearrange(arr, "b c h w -> c h (b w)")
print(arr.shape, arr1.shape)

if MAIN:
    display_array_as_img(arr1)

# %%
arr2 = einops.repeat(arr[0], "c h w -> c (2 h) w")

if MAIN:
    display_array_as_img(arr2)

# %%
inter = einops.rearrange(arr[:2], "b c h w -> c (b h) w")
arr3 = einops.repeat(inter, "c bh w -> c bh (2 w)")
# one-liner: einops.repeat(arr[:2], "b c h w -> c (b h) (2 w)") 

if MAIN:
        display_array_as_img(arr3)

# %% 
arr4 = einops.repeat(arr[0], "c h w -> c (h 2) w")

if MAIN:
    display_array_as_img(arr4)

# %% 
arr5 = einops.repeat(arr[0], "c h w -> h (c w)")
# einops.rearrange also works here

if MAIN:
    display_array_as_img(arr5)

# %% 
arr6 = einops.rearrange(arr, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=2)

if MAIN:
    display_array_as_img(arr6)

# %% 
arr7 = einops.reduce(arr, "b c h w -> h (b w)", 'max')

if MAIN:
    display_array_as_img(arr7)

# %% 
arr8 = einops.reduce(arr.astype(float), "b c h w -> h w", 'min')

if MAIN:
    display_array_as_img(arr8)

# %% 
arr9 = einops.rearrange(arr[1], "c h w -> c w h")

if MAIN:
    display_array_as_img(arr9)

# %% 
arr10 = einops.reduce(arr, "(b1 b2) c (h 2) (w 2) -> c (b1 h) (b2 w)", 'max', b1=2)

if MAIN:
    display_array_as_img(arr10)


###### EINSUM

# %%
def einsum_trace(mat: np.ndarray):
    '''
    Returns the same as `np.trace`. i.e. the sum of diagonals of the array
    '''
    return einops.einsum(mat, np.eye(mat.shape[-1]), "i i, i i ->")

def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat` is a 2D array and `vec` is 1D.
    '''
    return einops.einsum(mat, vec, "j i, i -> j")

def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat1` and `mat2` are both 2D arrays.
    '''
    return einops.einsum(mat1, mat2, "i k, k j -> i j")

def einsum_inner(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.inner`.
    '''
    return einops.einsum(vec1, vec2, "i, i ->")

def einsum_outer(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.outer`.
    '''
    return einops.einsum(vec1, vec2, "i, j -> i j")


if MAIN:
    tests.test_einsum_trace(einsum_trace)
    tests.test_einsum_mv(einsum_mv)
    tests.test_einsum_mm(einsum_mm)
    tests.test_einsum_inner(einsum_inner)
    tests.test_einsum_outer(einsum_outer)

