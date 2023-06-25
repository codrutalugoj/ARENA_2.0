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


if MAIN:
	reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)


# %%
from transformer import Config, DemoTransformer, get_log_probs

model_cfg = Config(
    debug=False, 
    d_model=256, 
    n_heads=4, 
    d_head=64, 
    d_mlp=1024, 
    n_layers=2, 
    n_ctx=256, 
    d_vocab=reference_gpt2.cfg.d_vocab
)
model = DemoTransformer(model_cfg)

# %% 
@dataclass
class TransformerTrainingArgs():
    batch_size = 8
    max_epochs = 1
    max_steps = 1000
    log_every = 10
    lr = 1e-3
    weight_decay = 1e-2
    log_dir: str = os.getcwd() + "/logs"
    log_name: str = "day1-transformer"
    run_name: Optional[str] = None
    log_every_n_steps: int = 1


args = TransformerTrainingArgs()

# %%
# CREATE DATA (contains only the first 10k entries from the Pile)
dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train").remove_columns("meta")
print(dataset)
print(dataset[0]['text'][:100])

# %%
tokenized_dataset = tokenize_and_concatenate(dataset, 
                                             reference_gpt2.tokenizer, 
                                             streaming=False, 
                                             max_length=model.cfg.n_ctx, 
                                             column_name="text", add_bos_token=True, 
                                             num_proc=4)
data_loader = DataLoader(tokenized_dataset, 
                         batch_size=args.batch_size, 
                         shuffle=True, 
                         num_workers=4, 
                         pin_memory=True)

# %%
first_batch = data_loader.dataset[:args.batch_size]

print(first_batch.keys())
print(first_batch['tokens'].shape)

# %%
class LitTransformer(pl.LightningModule):
      def __init__(self, 
                   args: TransformerTrainingArgs, 
                   model: DemoTransformer, 
                   data_loader: DataLoader):
            super().__init__()
            self.model = model
            self.cfg = model.cfg
            self.args = args
            self.data_loader = data_loader
            self.optimizer = self.configure_optimizers
    
      def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_vocab"]:
            logits = self.model(tokens)
            return logits
      
      def training_step(self, batch: Dict[str, Tensor], batch_idk: int) -> Float[Tensor, ""]:
            # compute training loss + additional metrics e.g. progress bar/logger
            tokens = batch["tokens"].to(device)
            logits = self.model(tokens)
            loss = -get_log_probs(logits, tokens).mean()
            self.log("train_loss", loss)

      def configure_optimizers(self):
            '''
            Set optimizers and learning-rate schedulers to use in the optimization.
            '''
            optimizer = t.optim.AdamW(self.model.parameters(),
                                      lr=self.args.lr,
                                      weight_decay=self.args.weight_decay)
            return optimizer
      
      def train_dataloader(self):
            return self.data_loader
      
# %%
litmodel = LitTransformer(args, model, data_loader)
logger = WandbLogger(save_dir=args.log_dir, project=args.log_name, name=args.run_name)

trainer = pl.Trainer(
    max_epochs=args.max_epochs,
    logger=logger,
    log_every_n_steps=args.log_every_n_steps
)
trainer.fit(model=litmodel, train_dataloaders=litmodel.data_loader)
wandb.finish()

# %%
'''
WHAT DOES THE MODEL LEARN? Algorithms:
1. random noise 
 -> model learns to predict each token in the vocab with uniform prob
 -> Q(x) = 1/d_vocab. 
 -> cross_entropy_loss = log(d_vocab)
'''
d_vocab = model.cfg.d_vocab

print(f"d_vocab = {d_vocab}")
print(f"Cross entropy loss on uniform distribution = {math.log(d_vocab)}")


'''
2. unigram frequencies
 -> it learns the frequencies of words in English
 -> e.g. tokens " and" or " the" are more frequent than others
 -> avg cross_entropy_loss = - sum_x (p_x log p_x)
'''
toks = tokenized_dataset[:]["tokens"].flatten()

d_vocab = model.cfg.d_vocab
freqs = t.bincount(toks, minlength=d_vocab)
probs = freqs.float() / freqs.sum()

distn = t.distributions.categorical.Categorical(probs=probs)
entropy = distn.entropy()

print(f"Entropy of training data = {entropy}")

'''
3. bigram frequencies
 -> i.e. frequency of pairs of adjacent tokens in the training data
 -> e.g. "I" and " am" have a higher bigram frequency than if they occurred independently
'''

'''
4. more advanced techniques
 -> trigrams
 -> induction heads
 -> fact memorization 
'''
 