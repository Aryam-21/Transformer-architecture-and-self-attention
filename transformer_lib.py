"""
transformer_lib.py
==================
Importable library: a decoder-only Transformer (from the paper "Attention Is All
You Need") with switchable positional encodings, trained on Tiny Shakespeare.

    sinusoidal : static sin/cos, added to embeddings
    learned    : trainable nn.Embedding, added to embeddings
    rope       : rotary embedding, applied to Q and K inside attention
    none       : no positional information (ablation baseline)

Use as a library (see benchmarks.ipynb):
    from transformer_lib import Config, load_data, train, evaluate_at_length

Or from the command line:
    python transformer_lib.py --pos rope
    python transformer_lib.py --pos all --max_iters 300 --device cpu   # quick test
"""

import argparse
import json
import math
import os
import time
import urllib.request
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
POS_TYPES = ["sinusoidal", "learned", "rope", "none"]


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
@dataclass
class Config:
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    block_size: int = 128        # training context length
    max_len_factor: int = 4      # fixed/rope buffers cover block_size * factor (for extrapolation tests)
    batch_size: int = 64
    dropout: float = 0.1
    lr: float = 1e-3
    warmup: int = 200
    max_iters: int = 3000
    eval_interval: int = 250
    eval_iters: int = 50
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_data(path="input.txt"):
    """Returns train_data, val_data, chars, stoi, itos (character-level, 90/10 split)."""
    if not os.path.exists(path):
        print("Downloading Tiny Shakespeare...")
        urllib.request.urlretrieve(DATA_URL, path)
    text = open(path, "r", encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], chars, stoi, itos


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ----------------------------------------------------------------------------
# Positional encodings
# ----------------------------------------------------------------------------
class SinusoidalPE(nn.Module):
    """Static: PE(pos,2i)=sin(pos/10000^(2i/d)), PE(pos,2i+1)=cos(...). No parameters."""

    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):  # (B, T, d_model)
        return x + self.pe[: x.size(1)]


class LearnedPE(nn.Module):
    """Learned: one trainable vector per position. Cannot go beyond `max_len`."""

    def __init__(self, d_model, max_len):
        super().__init__()
        self.max_len = max_len
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, x):
        T = x.size(1)
        if T > self.max_len:
            raise ValueError(f"Learned PE supports at most {self.max_len} positions, got {T}")
        return x + self.emb(torch.arange(T, device=x.device))


class RotaryPE(nn.Module):
    """RoPE: rotate Q and K by a position-dependent angle -> scores depend on relative distance."""

    def __init__(self, head_dim, max_len):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE needs an even head_dim"
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = torch.outer(torch.arange(max_len).float(), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k):  # (B, heads, T, head_dim)
        T = q.size(2)
        cos, sin = self.cos[:T], self.sin[:T]
        return q * cos + self._rotate_half(q) * sin, k * cos + self._rotate_half(k) * sin


# ----------------------------------------------------------------------------
# Transformer blocks
# ----------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention with h heads and a causal mask."""

    def __init__(self, d_model, n_heads, max_len, dropout, rope=None):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.rope = rope
        self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len)).bool())

    def forward(self, x):
        B, T, C = x.shape
        q = self.W_q(x).view(B, T, self.h, self.dk).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.h, self.dk).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.h, self.dk).transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope(q, k)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)       # MatMul + Scale
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))  # Mask
        weights = self.drop(F.softmax(scores, dim=-1))                 # SoftMax
        out = (weights @ v).transpose(1, 2).contiguous().view(B, T, C)  # MatMul + Concat
        return self.W_o(out)                                           # Linear


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Decoder layer, pre-LayerNorm variant: x = x + Sublayer(LN(x))."""

    def __init__(self, d_model, n_heads, d_ff, max_len, dropout, rope):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, max_len, dropout, rope)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, pos_type, cfg: Config):
        super().__init__()
        if pos_type not in POS_TYPES:
            raise ValueError(f"pos_type must be one of {POS_TYPES}")
        self.pos_type, self.d_model, self.block_size = pos_type, cfg.d_model, cfg.block_size
        max_len = cfg.block_size * cfg.max_len_factor
        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)

        self.pos_enc, rope = None, None
        if pos_type == "sinusoidal":
            self.pos_enc = SinusoidalPE(cfg.d_model, max_len)
        elif pos_type == "learned":
            self.pos_enc = LearnedPE(cfg.d_model, cfg.block_size)  # only trained positions exist
        elif pos_type == "rope":
            rope = RotaryPE(cfg.d_model // cfg.n_heads, max_len)

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [Block(cfg.d_model, cfg.n_heads, cfg.d_ff, max_len, cfg.dropout, rope) for _ in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx) * math.sqrt(self.d_model)
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size :])
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


# ----------------------------------------------------------------------------
# Training and benchmarking
# ----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg: Config):
    model.eval()
    out = {}
    for name, d in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(cfg.eval_iters)
        for i in range(cfg.eval_iters):
            x, y = get_batch(d, cfg.block_size, cfg.batch_size, cfg.device)
            losses[i] = model(x, y)[1].item()
        out[name] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def evaluate_at_length(model, data, length, cfg: Config, iters=30, batch_size=16):
    """Validation loss on sequences of a given length (may be longer than the training block_size).
    Returns NaN if the model cannot handle that length (e.g. learned PE beyond block_size)."""
    model.eval()
    losses = []
    try:
        for _ in range(iters):
            x, y = get_batch(data, length, batch_size, cfg.device)
            losses.append(model(x, y)[1].item())
    except ValueError:
        return float("nan")
    return sum(losses) / len(losses)


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def train(pos_type, cfg: Config, train_data, val_data, vocab_size, itos, verbose=True, sample_tokens=300):
    """Train one model. Returns a dict with the model, loss history, params, timing, and a text sample."""
    torch.manual_seed(cfg.seed)
    model = TransformerLM(vocab_size, pos_type, cfg).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"\n=== {pos_type} | {n_params/1e6:.2f}M params | device={cfg.device} ===")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / cfg.warmup, 0.5 * (1 + math.cos(math.pi * min(s, cfg.max_iters) / cfg.max_iters)))
    )

    hist = {"step": [], "train": [], "val": []}
    train_time = 0.0
    for step in range(cfg.max_iters + 1):
        if step % cfg.eval_interval == 0:
            l = estimate_loss(model, train_data, val_data, cfg)
            hist["step"].append(step); hist["train"].append(l["train"]); hist["val"].append(l["val"])
            if verbose:
                print(f"step {step:5d} | train {l['train']:.4f} | val {l['val']:.4f}")
        if step == cfg.max_iters:
            break
        x, y = get_batch(train_data, cfg.block_size, cfg.batch_size, cfg.device)
        _sync(cfg.device); t0 = time.perf_counter()
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        _sync(cfg.device); train_time += time.perf_counter() - t0

    model.eval()
    ctx = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)
    sample = "".join(itos[i] for i in model.generate(ctx, sample_tokens)[0].tolist())
    return {
        "pos_type": pos_type, "model": model, "params": n_params, "history": hist,
        "train_time_s": train_time, "sec_per_step": train_time / cfg.max_iters, "sample": sample,
    }


def save_results(results, path="results.json"):
    """Save everything except the model objects."""
    slim = {k: {kk: vv for kk, vv in v.items() if kk != "model"} for k, v in results.items()}
    json.dump(slim, open(path, "w"), indent=2)


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pos", default="all", choices=POS_TYPES + ["all"])
    for name, default in asdict(Config()).items():
        p.add_argument(f"--{name}", type=type(default), default=default)
    args = vars(p.parse_args())
    pos = args.pop("pos")
    cfg = Config(**args)

    train_data, val_data, chars, stoi, itos = load_data()
    print(f"vocab={len(chars)} | train={len(train_data):,} | val={len(val_data):,}")
    results = {}
    for t in (POS_TYPES if pos == "all" else [pos]):
        results[t] = train(t, cfg, train_data, val_data, len(chars), itos)
        print(f"--- sample ({t}) ---\n{results[t]['sample']}\n")
    print("Final validation loss:")
    for t, r in results.items():
        print(f"  {t:11s} {r['history']['val'][-1]:.4f}")
    save_results(results)


if __name__ == "__main__":
    main()