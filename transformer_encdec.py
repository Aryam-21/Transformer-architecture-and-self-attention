"""
transformer_encdec.py  (v2)
===========================
Full ENCODER-DECODER Transformer (Figure 1 of "Attention Is All You Need") trained on
sequence-to-sequence tasks built from Tiny Shakespeare, with switchable positional encodings.

Tasks (source -> target, fixed-length character sequences):
    reverse      : target = source reversed (needs cross-attention AND precise position info)
    continuation : source = 64 chars, target = the next 64 chars of the text

Positional encodings (applied on the source AND target sides):
    sinusoidal : static sin/cos added to embeddings
    learned    : trainable table (separate for source and target); rows beyond the training
                 length exist but are untrained
    rope       : rotary, applied to Q/K in self-attention (not in cross-attention)
    none       : no position info (baseline)

What is new in v2
    * generate_eval(): decodes the SAME fixed validation examples for every model (fair comparison)
    * generate(): greedy or temperature sampling; BOS can never be emitted
    * continue_text(): chunked long-text generation for the continuation task
    * per_position_accuracy(), decode_text(), sequence_metrics() helpers

Library use:
    from transformer_encdec import Config, load_data, train, generate_eval, decode_text

Command line:
    python transformer_encdec.py --task reverse --pos all
    python transformer_encdec.py --task continuation --pos rope --max_iters 300 --device cpu
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
TASKS = ["reverse", "continuation"]


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
@dataclass
class Config:
    task: str = "reverse"
    d_model: int = 256
    n_heads: int = 4
    n_enc_layers: int = 3        # paper uses 6; 3 trains faster
    n_dec_layers: int = 3
    d_ff: int = 1024
    src_len: int = 64
    tgt_len: int = 64
    max_len_factor: int = 4      # fixed/rope/learned tables cover length * factor
    batch_size: int = 64
    dropout: float = 0.1
    lr: float = 1e-3
    warmup: int = 200
    max_iters: int = 3000
    eval_interval: int = 250
    eval_iters: int = 30
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


def fixed_indices(data, cfg, n, seed=0):
    """Deterministic start positions, so every model is evaluated on exactly the same examples."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(len(data) - cfg.src_len - cfg.tgt_len - 1, (n,), generator=g)


def get_batch(data, task, src_len, tgt_len, batch_size, device, bos_id, ix=None):
    """Returns (src, dec_in, tgt).  dec_in = [BOS] + tgt[:-1]  ("outputs shifted right").
    If `ix` (start positions) is given, those exact examples are used instead of random ones."""
    if ix is None:
        ix = torch.randint(len(data) - src_len - tgt_len - 1, (batch_size,))
    batch_size = len(ix)
    src = torch.stack([data[i : i + src_len] for i in ix])
    if task == "reverse":
        assert src_len == tgt_len, "reverse task needs src_len == tgt_len"
        tgt = src.flip(1)
    elif task == "continuation":
        tgt = torch.stack([data[i + src_len : i + src_len + tgt_len] for i in ix])
    else:
        raise ValueError(f"task must be one of {TASKS}")
    bos = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    dec_in = torch.cat([bos, tgt[:, :-1]], dim=1)
    return src.to(device), dec_in.to(device), tgt.to(device)


def decode_text(t, itos, keep_newlines=False):
    """1-D tensor of ids -> string. Newlines become a pilcrow so one character = one column."""
    s = "".join(itos.get(int(i), "?") for i in t)
    return s if keep_newlines else s.replace("\n", "\u00b6")


# ----------------------------------------------------------------------------
# Positional encodings
# ----------------------------------------------------------------------------
class SinusoidalPE(nn.Module):
    """Static sin/cos, added to embeddings. No parameters."""

    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class LearnedPE(nn.Module):
    """
    One trainable vector per position.
    trained_len : positions that actually receive gradients during training
    alloc_len   : rows allocated in the table (>= trained_len). Rows beyond trained_len are NEVER
                  trained, so they keep their random initial values (untrained noise).
    strict=True : raise an error beyond trained_len instead of returning those untrained rows.
    """

    def __init__(self, d_model, trained_len, alloc_len=None, strict=False):
        super().__init__()
        self.trained_len, self.strict = trained_len, strict
        self.emb = nn.Embedding(alloc_len or trained_len, d_model)

    def forward(self, x):
        T = x.size(1)
        if T > self.emb.num_embeddings:
            raise ValueError(f"Learned PE table has {self.emb.num_embeddings} rows, got length {T}")
        if self.strict and T > self.trained_len:
            raise ValueError(f"Learned PE trained for {self.trained_len} positions, got {T}")
        return x + self.emb(torch.arange(T, device=x.device))


class RotaryPE(nn.Module):
    """RoPE: rotate Q and K by a position-dependent angle (relative position in the dot product)."""

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

    def forward(self, q, k):
        T = q.size(2)
        cos, sin = self.cos[:T], self.sin[:T]
        return q * cos + self._rotate_half(q) * sin, k * cos + self._rotate_half(k) * sin


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """
    Scaled dot-product multi-head attention. Queries come from x_q; keys and values from x_kv.
        self-attention  : x_kv is x_q
        cross-attention : x_kv is the encoder output (memory)
    causal=True adds the look-ahead mask used in the decoder's masked self-attention.
    """

    def __init__(self, d_model, n_heads, dropout, max_len, causal=False, rope=None):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk, self.causal, self.rope = n_heads, d_model // n_heads, causal, rope
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        if causal:
            self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len)).bool())

    def forward(self, x_q, x_kv):
        B, Tq, C = x_q.shape
        Tk = x_kv.size(1)
        q = self.W_q(x_q).view(B, Tq, self.h, self.dk).transpose(1, 2)
        k = self.W_k(x_kv).view(B, Tk, self.h, self.dk).transpose(1, 2)
        v = self.W_v(x_kv).view(B, Tk, self.h, self.dk).transpose(1, 2)
        if self.rope is not None:                      # self-attention only
            q, k = self.rope(q, k)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)               # MatMul + Scale
        if self.causal:
            scores = scores.masked_fill(~self.mask[:Tq, :Tk], float("-inf"))  # Mask
        weights = self.drop(F.softmax(scores, dim=-1))                        # SoftMax
        out = (weights @ v).transpose(1, 2).contiguous().view(B, Tq, C)       # MatMul + Concat
        return self.W_o(out)                                                  # Linear


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    """Self-attention (bidirectional) -> Feed Forward. Pre-LayerNorm residual blocks."""

    def __init__(self, cfg, max_len, rope):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout, max_len, causal=False, rope=rope)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.attn(h, h)
        return x + self.ff(self.ln2(x))


class DecoderLayer(nn.Module):
    """Masked self-attention -> Cross-attention (to encoder) -> Feed Forward."""

    def __init__(self, cfg, max_len, rope):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout, max_len, causal=True, rope=rope)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.cross_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout, max_len, causal=False, rope=None)
        self.ln3 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x, memory):
        h = self.ln1(x)
        x = x + self.self_attn(h, h)                      # masked: can't see future target tokens
        x = x + self.cross_attn(self.ln2(x), memory)      # Q from decoder, K/V from encoder
        return x + self.ff(self.ln3(x))


class EncoderDecoderTransformer(nn.Module):
    def __init__(self, vocab_size, pos_type, cfg: Config):
        """vocab_size = number of real characters. One extra id (= vocab_size) is the BOS token."""
        super().__init__()
        if pos_type not in POS_TYPES:
            raise ValueError(f"pos_type must be one of {POS_TYPES}")
        self.pos_type, self.d_model, self.bos_id = pos_type, cfg.d_model, vocab_size
        V = vocab_size + 1
        max_len = max(cfg.src_len, cfg.tgt_len) * cfg.max_len_factor

        self.tok_emb = nn.Embedding(V, cfg.d_model)        # shared by source, target, and output head
        self.pos_src = self.pos_tgt = None
        rope = None
        if pos_type == "sinusoidal":
            self.pos_src, self.pos_tgt = SinusoidalPE(cfg.d_model, max_len), SinusoidalPE(cfg.d_model, max_len)
        elif pos_type == "learned":
            self.pos_src = LearnedPE(cfg.d_model, cfg.src_len, alloc_len=max_len)
            self.pos_tgt = LearnedPE(cfg.d_model, cfg.tgt_len, alloc_len=max_len)
        elif pos_type == "rope":
            rope = RotaryPE(cfg.d_model // cfg.n_heads, max_len)

        self.drop = nn.Dropout(cfg.dropout)
        self.enc_layers = nn.ModuleList([EncoderLayer(cfg, max_len, rope) for _ in range(cfg.n_enc_layers)])
        self.enc_ln = nn.LayerNorm(cfg.d_model)
        self.dec_layers = nn.ModuleList([DecoderLayer(cfg, max_len, rope) for _ in range(cfg.n_dec_layers)])
        self.dec_ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, V, bias=False)
        self.head.weight = self.tok_emb.weight             # weight tying (as in the paper)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def encode(self, src):
        x = self.tok_emb(src) * math.sqrt(self.d_model)
        if self.pos_src is not None:
            x = self.pos_src(x)
        x = self.drop(x)
        for layer in self.enc_layers:
            x = layer(x)
        return self.enc_ln(x)                              # "memory"

    def decode(self, dec_in, memory):
        x = self.tok_emb(dec_in) * math.sqrt(self.d_model)
        if self.pos_tgt is not None:
            x = self.pos_tgt(x)
        x = self.drop(x)
        for layer in self.dec_layers:
            x = layer(x, memory)
        return self.head(self.dec_ln(x))                   # Linear (softmax lives in the loss)

    def forward(self, src, dec_in, targets=None):
        logits = self.decode(dec_in, self.encode(src))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, src, length, temperature=0.0):
        """Generate `length` tokens one at a time, feeding each prediction back in.
        temperature=0 -> greedy (argmax); temperature>0 -> sample. BOS is never emitted."""
        memory = self.encode(src)
        ys = torch.full((src.size(0), 1), self.bos_id, dtype=torch.long, device=src.device)
        for _ in range(length):
            logits = self.decode(ys, memory)[:, -1]
            logits[:, self.bos_id] = float("-inf")
            if temperature > 0:
                nxt = torch.multinomial(F.softmax(logits / temperature, dim=-1), 1)
            else:
                nxt = logits.argmax(-1, keepdim=True)
            ys = torch.cat([ys, nxt], dim=1)
        return ys[:, 1:]

    def greedy_decode(self, src, length):
        return self.generate(src, length, 0.0)


# ----------------------------------------------------------------------------
# Evaluation and generation
# ----------------------------------------------------------------------------
@torch.no_grad()
def _eval(model, data, cfg, src_len, tgt_len, iters, batch_size):
    model.eval()
    losses, accs = [], []
    for _ in range(iters):
        src, dec_in, tgt = get_batch(data, cfg.task, src_len, tgt_len, batch_size, cfg.device, model.bos_id)
        logits, loss = model(src, dec_in, tgt)
        losses.append(loss.item())
        accs.append((logits.argmax(-1) == tgt).float().mean().item())
    return sum(losses) / len(losses), sum(accs) / len(accs)


def estimate_loss(model, train_data, val_data, cfg: Config):
    """Teacher-forced loss and token accuracy on train and val."""
    out = {}
    for name, d in (("train", train_data), ("val", val_data)):
        out[name], out[name + "_acc"] = _eval(model, d, cfg, cfg.src_len, cfg.tgt_len, cfg.eval_iters, cfg.batch_size)
    model.train()
    return out


def evaluate_at_length(model, data, length, cfg: Config, iters=20, batch_size=16):
    """Loss/accuracy on sequences of a different length. NaN if the model cannot handle that length."""
    max_len = max(cfg.src_len, cfg.tgt_len) * cfg.max_len_factor
    if length > max_len:
        return {"loss": float("nan"), "acc": float("nan")}
    try:
        loss, acc = _eval(model, data, cfg, length, length, iters, batch_size)
    except ValueError:                                   # e.g. learned PE with strict=True
        return {"loss": float("nan"), "acc": float("nan")}
    return {"loss": loss, "acc": acc}


@torch.no_grad()
def generate_eval(model, data, cfg: Config, n=512, seed=0, temperature=0.0):
    """Decode a FIXED set of validation examples (identical for every model when n and seed match).
    Returns (src, tgt, pred) as CPU tensors of token ids."""
    model.eval()
    ix = fixed_indices(data, cfg, n, seed)
    src, _, tgt = get_batch(data, cfg.task, cfg.src_len, cfg.tgt_len, n, cfg.device, model.bos_id, ix=ix)
    pred = model.generate(src, tgt.size(1), temperature)
    return src.cpu(), tgt.cpu(), pred.cpu()


def sequence_metrics(pred, tgt):
    """Per-character accuracy and whole-sequence exact match."""
    m = pred == tgt
    return {"char_acc": m.float().mean().item(), "exact_match": m.all(1).float().mean().item()}


def per_position_accuracy(pred, tgt):
    """Accuracy at each target position (mean over examples)."""
    return (pred == tgt).float().mean(0)


def greedy_accuracy(model, data, cfg: Config, n=512, seed=0):
    """Free-running greedy decoding on a fixed set of validation examples."""
    _, tgt, pred = generate_eval(model, data, cfg, n=n, seed=seed, temperature=0.0)
    return sequence_metrics(pred, tgt)


@torch.no_grad()
def continue_text(model, prompt, cfg: Config, n_chunks=4, temperature=0.8):
    """Chunked long-text generation (continuation task): each predicted chunk is appended to the
    context, and the last `src_len` characters become the next source."""
    model.eval()
    src = prompt.view(1, -1).to(cfg.device)
    chunks = []
    for _ in range(n_chunks):
        pred = model.generate(src, cfg.tgt_len, temperature)
        chunks.append(pred[0].cpu())
        src = torch.cat([src, pred], dim=1)[:, -cfg.src_len :]
    return torch.cat(chunks)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def train(pos_type, cfg: Config, train_data, val_data, vocab_size, itos=None, verbose=True):
    """Train one model. Returns dict with model, history, params, timing, greedy metrics."""
    torch.manual_seed(cfg.seed)
    model = EncoderDecoderTransformer(vocab_size, pos_type, cfg).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"\n=== {cfg.task} | {pos_type} | {n_params/1e6:.2f}M params | {cfg.max_iters} steps | device={cfg.device} ===")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / cfg.warmup, 0.5 * (1 + math.cos(math.pi * min(s, cfg.max_iters) / cfg.max_iters)))
    )

    hist = {"step": [], "train_loss": [], "val_loss": [], "val_acc": []}
    train_time = 0.0
    for step in range(cfg.max_iters + 1):
        if step % cfg.eval_interval == 0 or step == cfg.max_iters:
            m = estimate_loss(model, train_data, val_data, cfg)
            hist["step"].append(step); hist["train_loss"].append(m["train"])
            hist["val_loss"].append(m["val"]); hist["val_acc"].append(m["val_acc"])
            if verbose:
                print(f"step {step:5d} | train {m['train']:.4f} | val {m['val']:.4f} | val tok-acc {m['val_acc']:.3f}")
        if step == cfg.max_iters:
            break
        src, dec_in, tgt = get_batch(train_data, cfg.task, cfg.src_len, cfg.tgt_len, cfg.batch_size, cfg.device, model.bos_id)
        _sync(cfg.device); t0 = time.perf_counter()
        _, loss = model(src, dec_in, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        _sync(cfg.device); train_time += time.perf_counter() - t0

    greedy = greedy_accuracy(model, val_data, cfg)
    if verbose:
        print(f"greedy decoding: char acc {greedy['char_acc']:.3f} | exact match {greedy['exact_match']:.3f}")
    return {
        "pos_type": pos_type, "model": model, "params": n_params, "history": hist,
        "train_time_s": train_time, "sec_per_step": train_time / cfg.max_iters, "greedy": greedy,
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
    print(f"task={cfg.task} | vocab={len(chars)} | train={len(train_data):,} | val={len(val_data):,}")
    results = {}
    for t in (POS_TYPES if pos == "all" else [pos]):
        results[t] = train(t, cfg, train_data, val_data, len(chars), itos)
        src, tgt, pred = generate_eval(results[t]["model"], val_data, cfg, n=1)
        print(f"source : {decode_text(src[0], itos)}\ntarget : {decode_text(tgt[0], itos)}\n"
              f"predict: {decode_text(pred[0], itos)}")
    print("\nSummary (final val loss | greedy char acc | exact match):")
    for t, r in results.items():
        print(f"  {t:11s} {r['history']['val_loss'][-1]:.4f} | {r['greedy']['char_acc']:.3f} | {r['greedy']['exact_match']:.3f}")
    save_results(results, f"results_{cfg.task}.json")


if __name__ == "__main__":
    main()