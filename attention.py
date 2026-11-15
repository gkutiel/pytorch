from math import sqrt

import torch
from torch import Tensor, nn
from torch.nn.functional import softmax


def attend(*,
           q: Tensor,
           k: Tensor,
           v: Tensor,
           mask: Tensor | None = None):
    # q (B, Q, D)
    # k (B, K, D)
    # v (B, K, V)
    # mask (B, Q, K)
    # scores (B, Q, K)
    scores = q @ k.transpose(-2, -1)
    scores = scores / sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf)

    # ws (B, Q, K)
    ws = softmax(scores, dim=-1)
    # res (B, Q, V)
    res = ws @ v
    return ws, res


class EncoderBlock(nn.Module):
    def __init__(self, *, d_model, d_ff, do=0.1):
        super().__init__()
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.ln1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model))

        self.ln2 = nn.LayerNorm(d_model)

        self.do = nn.Dropout(do)

    def forward(self, *, x):
        # x (B, T, D)

        # ATTENTION
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        _, out = attend(q=q, k=k, v=v)
        out = self.ln1(x + self.do(out))

        # FF
        out = self.ln2(out + self.do(self.ff(out)))

        return out
