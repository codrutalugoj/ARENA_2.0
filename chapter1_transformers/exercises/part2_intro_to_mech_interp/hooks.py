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
##############################
# Example of hook function used to modify attention pattern/activation at a certain layer:
'''
def hook_function(
    attn_pattern: Float[Tensor, "batch heads seqQ seqK"],
    hook: HookPoint
) -> Float[Tensor, "batch heads seqQ seqK"]:

    # modify attn_pattern (can be inplace)
    return attn_pattern

# Running a model with hooks:
# this adds multiple hooks at once
loss = model.run_with_hooks(
    tokens, 
    return_type="loss",
    fwd_hooks=[
        ('blocks.0.attn.hook_pattern', hook_function),
        ('blocks.1.attn.hook_pattern', hook_function)
    ]
)

# or using a name pattern:
loss = model.run_with_hooks(
    tokens,
    return_type="loss",
    fwd_hooks=[
        (lambda name: name.endswith("pattern"), hook_function)
    ]
)
'''
##############################

# %%
from induction_heads import generate_repeated_tokens

seq_len = 50
batch = 10
rep_tokens_10 = generate_repeated_tokens(model, seq_len, batch)

# We make a tensor to store the induction score for each head.
# We put it on the model's device to avoid needing to move things between the GPU and CPU, which can be slow.
induction_score_store = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)


def induction_score_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    '''
    Calculates the induction score, and stores it in the [layer, head] position of the `induction_score_store` tensor.
    '''
    batch, h_index, dest_pos, source_pos = pattern.shape
    print(f"{pattern.shape=} {hook.layer()=}")

    score = pattern.diagonal(offset=-(dest_pos//2)+1, dim1=-2, dim2=-1)
    score = score.mean(dim=(0, -1)) # mean over batch (dim=0) and seq _len (dim=-1)

    induction_score_store[hook.layer(), :] = score


pattern_hook_names_filter = lambda name: name.endswith("pattern")

# Run with hooks (this is where we write to the `induction_score_store` tensor`)
model.run_with_hooks(
    rep_tokens_10, 
    return_type=None, # For efficiency, we don't need to calculate the logits
    fwd_hooks=[(
        pattern_hook_names_filter,
        induction_score_hook
    )]
)

# Plot the induction scores for each head in each layer
imshow(
    induction_score_store, 
    labels={"x": "Head", "y": "Layer"}, 
    title="Induction Score by Head", 
    text_auto=".2f",
    width=900, height=400
)


# %%
def visualize_pattern_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    print("Layer: ", hook.layer())
    display(
        cv.attention.attention_patterns(
            tokens=gpt2_small.to_str_tokens(rep_tokens_10[0]), 
            attention=pattern.mean(0)
        )
    )


# YOUR CODE HERE - find induction heads in gpt2_small
gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")

induction_score_store = t.zeros((gpt2_small.cfg.n_layers, gpt2_small.cfg.n_heads), device=gpt2_small.cfg.device)


gpt2_small.run_with_hooks(
    rep_tokens_10,
    return_type=None,
    fwd_hooks=[(pattern_hook_names_filter,
               induction_score_hook)]
)

# Plot the induction scores for each head in each layer
imshow(
    induction_score_store, 
    labels={"x": "Head", "y": "Layer"}, 
    title="Induction Score by Head", 
    text_auto=".2f",
    width=900, height=400
)

gpt2_small.run_with_hooks(
    rep_tokens_10,
    return_type=None,
    fwd_hooks=[(pattern_hook_names_filter,
               visualize_pattern_hook)]
)


# %%
# Interp tools: finding component role in logits
def logit_attribution(
    embed: Float[Tensor, "seq d_model"],
    l1_results: Float[Tensor, "seq nheads d_model"],
    l2_results: Float[Tensor, "seq nheads d_model"],
    W_U: Float[Tensor, "d_model d_vocab"],
    tokens: Int[Tensor, "seq"]
) -> Float[Tensor, "seq-1 n_components"]:
    '''
    Inputs:
        embed: the embeddings of the tokens (i.e. token + position embeddings)
        l1_results: the outputs of the attention heads at layer 1 (with head as one of the dimensions)
        l2_results: the outputs of the attention heads at layer 2 (with head as one of the dimensions)
        W_U: the unembedding matrix
        tokens: the token ids of the sequence

    Returns:
        Tensor of shape (seq_len-1, n_components)
        represents the concatenation (along dim=-1) of logit attributions from:
            the direct path (seq-1,1)
            layer 0 logits (seq-1, n_heads)
            layer 1 logits (seq-1, n_heads)
        so n_components = 1 + 2*n_heads
    '''
    W_U_correct_tokens = W_U[:, tokens[1:]]
    
    embed_contribution = einops.einsum(W_U_correct_tokens, embed[:-1], "d_model seq, seq d_model -> seq").unsqueeze(-1)
    l1_contribution = einops.einsum(W_U_correct_tokens, l1_results[:-1], "d_model seq, seq nheads d_model -> seq nheads")
    l2_contribution = einops.einsum(W_U_correct_tokens, l2_results[:-1], "d_model seq, seq nheads d_model -> seq nheads")

    return t.cat([embed_contribution, l1_contribution, l2_contribution], dim=-1)


text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
logits, cache = model.run_with_cache(text, remove_batch_dim=True)
str_tokens = model.to_str_tokens(text)
tokens = model.to_tokens(text)

with t.inference_mode():
    embed = cache["embed"]
    l1_results = cache["result", 0]
    l2_results = cache["result", 1]
    logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])
    # Uses fancy indexing to get a len(tokens[0])-1 length tensor, where the kth entry is the predicted logit for the correct k+1th token
    correct_token_logits = logits[0, t.arange(len(tokens[0]) - 1), tokens[0, 1:]]
    t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-3, rtol=0)
    print("Tests passed!")


# %%
embed = cache["embed"]
l1_results = cache["result", 0]
l2_results = cache["result", 1]
logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])

plot_logit_attribution(model, logit_attr, tokens)


# %% 
# Logit attribution for induction heads (attention-only model)
from induction_heads import run_and_cache_model_repeated_tokens

batch = 1
seq_len = 50
(rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch)
rep_cache.remove_batch_dim()

embed = rep_cache["embed"]
l1_results = rep_cache["result", 0]
l2_results = rep_cache["result", 1]
first_half_tokens = rep_tokens[0, : 1 + seq_len]
second_half_tokens = rep_tokens[0, seq_len:]


# YOUR CODE HERE - define `first_half_logit_attr` and `second_half_logit_attr`
first_half_logit_attr = logit_attribution(embed[:seq_len+1], l1_results[:seq_len+1], l2_results[:seq_len+1], model.W_U, first_half_tokens)
second_half_logit_attr = logit_attribution(embed[seq_len:], l1_results[seq_len:], l2_results[seq_len:], model.W_U, second_half_tokens)

assert first_half_logit_attr.shape == (seq_len, 2*model.cfg.n_heads + 1)
assert second_half_logit_attr.shape == (seq_len, 2*model.cfg.n_heads + 1)

plot_logit_attribution(model, first_half_logit_attr, first_half_tokens, "Logit attribution (first half of repeated sequence)")
plot_logit_attribution(model, second_half_logit_attr, second_half_tokens, "Logit attribution (second half of repeated sequence)")


# %%
##### ABLATIONS 
def head_ablation_hook(
    v: Float[Tensor, "batch seq n_heads d_head"],
    hook: HookPoint,
    head_index_to_ablate: int
) -> Float[Tensor, "batch seq n_heads d_head"]:
    
    v[:, :, head_index_to_ablate, :] = 0.0
    return v



def cross_entropy_loss(logits, tokens):
    '''
    Computes the mean cross entropy between logits (the model's prediction) and tokens (the true values).

    (optional, you can just use return_type="loss" instead.)
    '''
    log_probs = F.log_softmax(logits, dim=-1)
    pred_log_probs = t.gather(log_probs[:, :-1], -1, tokens[:, 1:, None])[..., 0]
    return -pred_log_probs.mean()


def get_ablation_scores(
    model: HookedTransformer, 
    tokens: Int[Tensor, "batch seq"]
) -> Float[Tensor, "n_layers n_heads"]:
    '''
    Returns a tensor of shape (n_layers, n_heads) containing the increase in cross entropy loss from ablating the output of each head.
    '''
    # Initialize an object to store the ablation scores
    ablation_scores = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)

    # Calculating loss without any ablation, to act as a baseline
    model.reset_hooks()
    logits = model(tokens, return_type="logits")
    loss_no_ablation = cross_entropy_loss(logits, tokens)

    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            # Use functools.partial to create a temporary hook function with the head number fixed
            temp_hook_fn = functools.partial(head_ablation_hook, head_index_to_ablate=head)
            # Run the model with the ablation hook
            ablated_logits = model.run_with_hooks(tokens, fwd_hooks=[
                (utils.get_act_name("v", layer), temp_hook_fn)
            ])
            # Calculate the logit difference
            loss = cross_entropy_loss(ablated_logits, tokens)
            # Store the result, subtracting the clean loss so that a value of zero means no change in loss
            ablation_scores[layer, head] = loss - loss_no_ablation

    return ablation_scores


ablation_scores = get_ablation_scores(model, rep_tokens)
tests.test_get_ablation_scores(ablation_scores, model, rep_tokens)

imshow(
    ablation_scores, 
    labels={"x": "Head", "y": "Layer", "color": "Logit diff"},
    title="Logit Difference After Ablating Heads", 
    text_auto=".2f",
    width=900, height=400
)

# %%
# WIP: NOT CORRECT
# Different ablation: 
# 1. ablate every head apart from the previous token head (layer 0, head 7) 
#    and the induction heads (layer 1, heads 4 and 10).

def get_ablation_scores_selection(
    model: HookedTransformer, 
    tokens: Int[Tensor, "batch seq"],
    heads_to_ablate: Int[Tensor, "n_layers n_heads"]
) -> Float[Tensor, "n_layers n_heads"]:
    '''
    Returns a tensor of shape (n_layers, n_heads) containing the increase in cross entropy loss from ablating the output of each head.
    '''
    # Initialize an object to store the ablation scores
    ablation_scores = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)

    # Calculating loss without any ablation, to act as a baseline
    model.reset_hooks()
    logits = model(tokens, return_type="logits")
    loss_no_ablation = cross_entropy_loss(logits, tokens)

    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            # ablate all heads except for 0.7, 1.4 and 1.10:
            if heads_to_ablate[layer, head].item() == 0:
                # Use functools.partial to create a temporary hook function with the head number fixed
                temp_hook_fn = functools.partial(head_ablation_hook, head_index_to_ablate=head)
                # Run the model with the ablation hook
                ablated_logits = model.run_with_hooks(tokens, fwd_hooks=[
                    (utils.get_act_name("v", layer), temp_hook_fn)
                ])
                # Calculate the logit difference
                loss = cross_entropy_loss(ablated_logits, tokens)
                # Store the result, subtracting the clean loss so that a value of zero means no change in loss
                ablation_scores[layer, head] = loss - loss_no_ablation

    return ablation_scores

heads_to_ablate_no_prevtoken_indhead = t.tensor(([[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], 
                                        [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1]]), dtype=t.bool)
ablation_scores_selection = get_ablation_scores_selection(model, rep_tokens, heads_to_ablate_no_prevtoken_indhead)

imshow(
    ablation_scores_selection, 
    labels={"x": "Head", "y": "Layer", "color": "Logit diff"},
    title="Logit Difference After Ablating Heads", 
    text_auto=".2f",
    width=900, height=400
)