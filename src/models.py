"""GPT-2 item policy and history-conditioned, two-head response predictor."""

import numpy as np
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_policy(vocab):
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=vocab,
            n_positions=32,
            n_ctx=32,
            n_embd=64,
            n_layer=2,
            n_head=4,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=None,
            use_cache=False,
        )
    )


def policy_logits(model, x):
    mask = x.ne(0)
    positions = (mask.long().cumsum(1) - 1).clamp(min=0)
    logits = model(input_ids=x, attention_mask=mask, position_ids=positions).logits[:, -1, :]
    return logits.masked_fill(torch.arange(logits.shape[-1], device=x.device)[None, :] < 2, -1e9)


def append_action(x, a):
    return torch.cat([x[:, 1:], a[:, None]], dim=1)


class RewardModel(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.embedding = nn.Embedding(vocab, 32, padding_idx=0)
        self.net = nn.Sequential(
            nn.Linear(97, 96),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x, a):
        emb = self.embedding(x)
        mask = x.ne(0)
        hist = (emb * mask[:, :, None]).sum(1) / mask.sum(1).clamp(min=1)[:, None]
        item = self.embedding(a)
        features = torch.cat([hist, item, hist * item, mask.sum(1, keepdim=True) / 32], 1)
        return self.net(features).sigmoid()


def ensemble(models, x, a):
    values = torch.stack([m(x, a) for m in models])
    return values.mean(0), values.std(0, unbiased=False)
