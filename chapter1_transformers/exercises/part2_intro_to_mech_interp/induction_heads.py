# %%
import os
import sys
import plotly.express as px
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import einops
from jaxtyping import Int, Float
from typing import List, Optional, Tuple
import functools
from tqdm import tqdm
from IPython.display import display
import webbrowser
import gdown
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import circuitsvis as cv

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part2_intro_to_mech_interp").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, hist, plot_comp_scores, plot_logit_attribution, plot_loss_difference
from part1_transformer_from_scratch.solutions import get_log_probs
import part2_intro_to_mech_interp.tests as tests

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

# %%
cfg = HookedTransformerConfig(
    d_model=768,
    d_head=64,
    n_heads=12,
    n_layers=2,
    n_ctx=2048,
    d_vocab=50278,
    attention_dir="causal",
    attn_only=True, # defaults to False
    tokenizer_name="EleutherAI/gpt-neox-20b", 
    seed=398,
    use_attn_result=True,
    normalization_type=None, # defaults to "LN", i.e. layernorm with weights & biases
    positional_embedding_type="shortformer"
)

# %%
# load weights
weights_dir = (section_dir / "attn_only_2L_half.pth").resolve()

if not weights_dir.exists():
    url = "https://drive.google.com/uc?id=1vcZLJnJoYKQs-2KOjkd6LvHZrkSdoxhu"
    output = str(weights_dir)
    gdown.download(url, output)


# %%
model = HookedTransformer(cfg)
pretrained_weights = t.load(weights_dir, map_location=device)
model.load_state_dict(pretrained_weights)

# %%
text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."

prompt_tokens = model.to_str_tokens(text)

logits, cache = model.run_with_cache(text, remove_batch_dim=True)

for l in range(cfg.n_layers):
    print("Layer ", l)
    display(cv.attention.attention_patterns(
        tokens=prompt_tokens, 
        attention=cache["pattern", l],
        attention_head_names=[f"L0H{i}" for i in range(cfg.n_heads)],
    ))

# %%


# %%
def current_attn_detector(cache: ActivationCache, threshold: float = 0.5) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
    '''
    out = [] 
    attn_idx = t.arange(cache["pattern", 0][0].shape[0])

    for l in range(cfg.n_layers):
        for head_num in range(cfg.n_heads): 
            token_attended = cache["pattern", l][head_num].argmax(dim=-1) 

            score = (token_attended == attn_idx).float().mean()
            #print(f"L{l}H{head_num}, score", score)
            if score > threshold:
                out.append(f"L{l}H{head_num}")

    return out

def prev_attn_detector(cache: ActivationCache, threshold: float = 0.7) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
    '''
    out = [] 
    attn_idx = t.arange(-1, cache["pattern", 0][0].shape[0] - 1)
    attn_idx[0] = 0

    for l in range(cfg.n_layers):
        for head_num in range(cfg.n_heads): 
            token_attended = cache["pattern", l][head_num].argmax(dim=-1) 

            score = (token_attended == attn_idx).float().mean()
            print(f"L{l}H{head_num}, score", score)
            if score > threshold:
                out.append(f"L{l}H{head_num}")

    return out

def first_attn_detector(cache: ActivationCache, threshold: float = 0.7) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads
    '''

    out = [] 
    for l in range(cfg.n_layers):
        for head_num in range(cfg.n_heads): 
            token_attended = cache["pattern", l][head_num].argmax(dim=-1) 
            score = (token_attended == 0).float().mean()
            #print(f"L{l}H{head_num}, score", score)
            if score > threshold:
                out.append(f"L{l}H{head_num}")

    return out


print("Heads attending to current token  = ", ", ".join(current_attn_detector(cache)))
print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))
print("Heads attending to first token    = ", ", ".join(first_attn_detector(cache)))


# %%
def generate_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch: int = 1
) -> Int[Tensor, "batch full_seq_len"]:
    '''
    Generates a sequence of repeated random tokens

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
    '''
    prefix = (t.ones(batch, 1) * model.tokenizer.bos_token_id).long()

    token_idx = t.randint(low=0, high=cfg.d_vocab, size=(batch, seq_len), dtype=t.int64)
    rep_tokens = t.cat([prefix, token_idx, token_idx], dim=-1).to(device)

    return rep_tokens

def run_and_cache_model_repeated_tokens(model: HookedTransformer, seq_len: int, batch: int = 1) -> Tuple[t.Tensor, t.Tensor, ActivationCache]:
    '''
    Generates a sequence of repeated random tokens, and runs the model on it, returning logits, tokens and cache

    Should use the `generate_repeated_tokens` function above

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
        rep_logits: [batch, 1+2*seq_len, d_vocab]
        rep_cache: The cache of the model run on rep_tokens
    '''
    rep_tokens = generate_repeated_tokens(model, seq_len, batch)
    rep_logits, rep_cache = model.run_with_cache(rep_tokens)

    return rep_tokens, rep_logits, rep_cache


seq_len = 50
batch = 1
(rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch)
rep_cache.remove_batch_dim()
rep_str = model.to_str_tokens(rep_tokens)
model.reset_hooks()
log_probs = get_log_probs(rep_logits, rep_tokens).squeeze()

print(f"Performance on the first half: {log_probs[:seq_len].mean():.3f}")
print(f"Performance on the second half: {log_probs[seq_len:].mean():.3f}")

plot_loss_difference(log_probs, rep_str, seq_len)

# %%

print(rep_tokens)
print()

for l in range(cfg.n_layers):
    print("Layer ", l)
    display(cv.attention.attention_patterns(
        tokens=rep_str, 
        attention=rep_cache["pattern", l],
        attention_head_names=[f"L{l}H{i}" for i in range(cfg.n_heads)],
    ))


# %%
def induction_attn_detector(cache: ActivationCache, threshold: float = 0.4) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be induction heads

    Remember - the tokens used to generate rep_cache are (bos_token, *rand_tokens, *rand_tokens)
    '''
    out = [] 
    n_induction_tokens = (cache["pattern", 0][0].shape[0])//2

    for l in range(cfg.n_layers):
        for head_num in range(cfg.n_heads): 
            token_attended = cache["pattern", l][head_num].diagonal(offset=-n_induction_tokens+1) 
            score = token_attended.mean()
            print(score)
            if score > threshold:
                out.append(f"L{l}H{head_num}")

    return out

print("Induction heads = ", ", ".join(induction_attn_detector(rep_cache)))