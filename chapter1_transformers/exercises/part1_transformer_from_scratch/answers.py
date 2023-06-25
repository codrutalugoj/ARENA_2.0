# %% 
import os; os.environ['ACCELERATE_DISABLE_RICH'] = "1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import sys
import einops
from dataclasses import dataclass
from transformer_lens import HookedTransformer
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
import torch as t
from torch import Tensor
import torch.nn as nn
import numpy as np
import math
from tqdm.notebook import tqdm
from typing import Tuple, List, Optional, Dict
from jaxtyping import Float, Int
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from collections import defaultdict
from rich.table import Table
from rich import print as rprint
import datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from pathlib import Path
import webbrowser

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part1_transformer_from_scratch").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow
# import part1_transformer_from_scratch.solutions as solutions

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == '__main__'

reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)


# %%
sorted_vocab = sorted(list(reference_gpt2.tokenizer.vocab.items()), key= lambda id: id[1])
print(sorted_vocab)

print(sorted_vocab[:20])
print()
print(sorted_vocab[250:270])
print()
print(sorted_vocab[990:1010])
print()
print(sorted_vocab[-20:])

# %% 
print(reference_gpt2.to_str_tokens("56873+3184623=123456789-1000000000"))


# %%
# Text Generation
reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"

# 1. Text -> tokens
# tokens have shape [batch, seq_len]
tokens = reference_gpt2.to_tokens(reference_text).to(device)
print(tokens)
print(tokens.shape)
print(reference_gpt2.to_str_tokens(tokens))

# 2. Map tokens to logits
# logits have shape [batch, seq_len, vocab_size]
logits, cache = reference_gpt2.run_with_cache(tokens)
print(logits.shape)

# 3. Logits -> ditribution
# probs have shape [batch, seq_len, vocab_size]
probs = t.softmax(logits, dim=-1)
print(probs.shape)

# 4. What's the most likely next token at each position?
most_likely_next_tokens = reference_gpt2.tokenizer.batch_decode(logits.argmax(dim=-1)[0])

print(list(zip(reference_gpt2.to_str_tokens(tokens), most_likely_next_tokens)))

# 4. Map distribution to a token
next_token = logits[0, -1].argmax(dim=-1)
next_char = reference_gpt2.to_string(next_token)
print(repr(next_char))

# %% 
# 5. Add the last result to the end of the sequence and re-run
print(f"Sequence so far:{reference_gpt2.to_string(tokens)[0]!r}")

for i in range(10):
    print(f"{tokens.shape[-1]+1}th char = {next_char!r}")
    # Define new input sequence, by appending the previously generated token
    tokens = t.cat([tokens, next_token[None, None]], dim=-1)
    # Pass our new sequence through the model, to get new output
    logits = reference_gpt2(tokens)
    # Get the predicted token at the end of our sequence
    next_token = logits[0, -1].argmax(dim=-1)
    # Decode and print the result
    next_char = reference_gpt2.to_string(next_token)


# %%
batch = 1
position = 35
d_model = 768
nheads = 12
n_layers = 12
d_mlp = 3072 # (= 4 * d_model)
d_head = 64#  (= d_model / nheads)


# %%
# Activation shapes
for activation_name, activation in cache.items():
    # Only print for first layer
    if ".0." in activation_name or "blocks" not in activation_name:
        print(f"{activation_name:30} {tuple(activation.shape)}")

# %%
# Parameters shapes
for name, param in reference_gpt2.named_parameters():
    # Only print for first layer
    if ".0." in name or "blocks" not in name:
        print(f"{name:18} {tuple(param.shape)}")

# %%
print(reference_gpt2.cfg)

@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    nheads: int = 12
    n_layers: int = 12


cfg = Config()
print(cfg)

# %%
# Simple tests:

def rand_float_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    random_input = t.randn(shape).to(device)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    if isinstance(output, tuple): output = output[0]
    print("Output shape:", output.shape, "\n")

def rand_int_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    random_input = t.randint(100, 1000, shape).to(device)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    if isinstance(output, tuple): output = output[0]
    print("Output shape:", output.shape, "\n")

def load_gpt2_test(cls, gpt2_layer, input):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    layer.load_state_dict(gpt2_layer.state_dict(), strict=False)
    print("Input shape:", input.shape)
    output = layer(input)
    if isinstance(output, tuple): output = output[0]
    print("Output shape:", output.shape)
    try: reference_output = gpt2_layer(input)
    except: reference_output = gpt2_layer(input, input, input)
    print("Reference output shape:", reference_output.shape, "\n")
    comparison = t.isclose(output, reference_output, atol=1e-4, rtol=1e-3)

    print(f"{comparison.sum()/comparison.numel():.2%} of the values are correct\n")

# %%
class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(self, residual: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        
        expected_x = residual.mean(dim=-1, keepdims=True)
        var_x = residual.var(dim=-1, keepdims=True, unbiased=False)

        norm_x = (residual - expected_x)/t.sqrt(var_x + cfg.layer_norm_eps)

        assert norm_x.mean() < 1e-6, f"Mean in LayerNorm is {norm_x.mean()}, but should be 0."
        assert (norm_x.var()).round() == 1, f"Variance in LayerNorm is {norm_x.var()}, but should be 1."
        
        return norm_x * self.w + self.b


rand_float_test(LayerNorm, [2, 4, 768])
load_gpt2_test(LayerNorm, reference_gpt2.ln_final, cache["resid_post", 11])


# %%
class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        return self.W_E[tokens]


rand_int_test(Embed, [2, 4])
load_gpt2_test(Embed, reference_gpt2.embed, tokens)

# %%
class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        indices = np.arange(tokens.shape[-1])

        return self.W_pos[indices]


rand_int_test(PosEmbed, [2, 4])
load_gpt2_test(PosEmbed, reference_gpt2.pos_embed, tokens)


# %%
class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.nheads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.nheads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.nheads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.nheads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.nheads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.nheads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.nheads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer("IGNORE", t.tensor(-1e5, dtype=t.float32, device=device))

    def forward(
        self, normalized_resid_pre: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:
        # Calculate key, query, value vectors 
        K = einops.einsum(normalized_resid_pre, 
                          self.W_K, 
                          "b seq_len d_model, nheads d_model d_head -> b seq_len nheads d_head") + self.b_K
        Q = einops.einsum(normalized_resid_pre, 
                          self.W_Q, 
                          "b seq_len d_model, nheads d_model d_head -> b seq_len nheads d_head") + self.b_Q

        # Compute attention scores by calculating the dot product for every key,query pair
        # query = destination
        # key = source
        attn_scores = einops.einsum(Q, K, "b seq_q nheads d_head, b seq_k nheads d_head -> b nheads seq_q seq_k")
        # scaling with sqrt(d_head)
        attn_scores = attn_scores / np.sqrt(self.cfg.d_head)
        attn_scores = self.apply_causal_mask(attn_scores)

        attn_pattern = t.softmax(attn_scores, dim=-1)

        # Computing values
        V = einops.einsum(normalized_resid_pre, 
                          self.W_V, 
                          "b seq_k d_model, nheads d_model d_head -> b seq_k nheads d_head") + self.b_V

        # weighted avg of values on seq_k dim
        z = einops.einsum(attn_pattern, V, 
                          "b nheads seq_q seq_k, b seq_k nheads d_head -> b seq_q nheads d_head")

        attn_out = einops.einsum(z, 
                            self.W_O, 
                            "b seq_q nheads d_head, nheads d_head d_model -> b seq_q d_model") + self.b_O

        return attn_out

    
    def apply_causal_mask(
        self, attn_scores: Float[Tensor, "batch nheads query_pos key_pos"]
    ) -> Float[Tensor, "batch nheads query_pos key_pos"]:
        '''
        Applies a causal mask to attention scores, and returns masked scores.
        '''
        #attn_scores2 = attn_scores

        mask = t.ones((attn_scores.size(-2), attn_scores.size(-1)), device=attn_scores.device)
        mask = t.triu(mask, diagonal=1).bool()

        attn_scores.masked_fill_(mask, self.IGNORE)

        
        return attn_scores


rand_float_test(Attention, [2, 4, 768])
load_gpt2_test(Attention, reference_gpt2.blocks[0].attn, cache["normalized", 0, "ln1"])


# %%
class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(
        self, normalized_resid_mid: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        pre = einops.einsum(normalized_resid_mid, 
                            self.W_in, 
                            "b pos d_model, d_model d_mlp -> b pos d_mlp") + self.b_in
        post = gelu_new(pre)
        
        mlp_out = einops.einsum(post, 
                            self.W_out,
                            "b pos d_mlp, d_mlp d_model -> b pos d_model") + self.b_out

        return mlp_out

rand_float_test(MLP, [2, 4, 768])
load_gpt2_test(MLP, reference_gpt2.blocks[0].mlp, cache["normalized", 0, "ln2"])


# %%
class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(
        self, resid_pre: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_model"]:
        
        resid_mid = self.attn(self.ln1(resid_pre)) + resid_pre
        resid_post = self.mlp(self.ln2(resid_mid)) + resid_mid

        return resid_post


rand_float_test(TransformerBlock, [2, 4, 768])
load_gpt2_test(TransformerBlock, reference_gpt2.blocks[0], cache["resid_pre", 0])


# %%
class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab), requires_grad=False))

    def forward(
        self, normalized_resid_final: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        return einops.einsum(normalized_resid_final, 
                            self.W_U, 
                            "b pos d_model, d_model d_vocab -> b pos d_vocab") + self.b_U


rand_float_test(Unembed, [2, 4, 768])
load_gpt2_test(Unembed, reference_gpt2.unembed, cache["ln_final.hook_normalized"])


# %%
class DemoTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_vocab"]:
        
        resid = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.blocks:
            resid = block(resid)
        logits = self.unembed(self.ln_final(resid))

        return logits

rand_int_test(DemoTransformer, [2, 4])
load_gpt2_test(DemoTransformer, reference_gpt2, tokens)


# %%
demo_gpt2 = DemoTransformer(Config(debug=False)).to(device)
demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)

demo_logits = demo_gpt2(tokens)


# %%
def get_log_probs(
        logits: Float[Tensor, "batch posn d_vocab"],
        tokens: Float[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
    
    log_probs = logits.log_softmax(dim=-1)

    # Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
    log_probs_for_tokens = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

    return log_probs_for_tokens

pred_log_probs = get_log_probs(demo_logits, tokens)
print(f"Avg cross entropy loss: {-pred_log_probs.mean():.4f}")
print(f"Avg cross entropy loss for uniform distribution: {math.log(demo_gpt2.cfg.d_vocab):4f}")
print(f"Avg probability assigned to correct token: {pred_log_probs.exp().mean():4f}")

test_string = '''The Total Perspective Vortex derives its picture of the whole Universe on the principle of'''
for i in tqdm(range(100)):
    test_tokens = reference_gpt2.to_tokens(test_string).to(device)
    demo_logits = demo_gpt2(test_tokens)
    test_string += reference_gpt2.tokenizer.decode(demo_logits[-1, -1].argmax())

print(test_string)