# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 20:48:13 2026

@author: farismismar
"""

"""
tiny_gpt.py  —  Minimal decoder-only transformer trained on Q&A pairs
======================================================================
Architecture mirrors Claude/GPT:
  token embedding + positional embedding
  → N × [LayerNorm + Causal Self-Attention (residual)
          LayerNorm + FFN with GELU        (residual)]
  → Final LayerNorm
  → Linear head (weight-tied to token embedding)

Training: next-token prediction on a QA-formatted text file.
Inference: generates until it hits a stop sequence ("\nQ:") so it
           returns exactly one answer without bleeding into the next question.

Requirements: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
"""

import math, os, sys  #, argparse
import matplotlib.pyplot as plt
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# ── Hyper-parameters ──────────────────────────────────────────────────────────
CONTEXT   = 128     # longer context to fit full QA pairs
D_MODEL   = 64      # wider embeddings for better recall
N_HEADS   = 4       # attention heads  (D_MODEL must be divisible by N_HEADS)
N_LAYERS  = 2       # two transformer blocks
FFN_DIM   = 128     # feed-forward hidden size
DROPOUT   = 0.1

EPOCHS    = 1200    # more epochs for the larger model to converge
LR        = 3e-3
BATCH     = 16

TRAIN_FILE  = "knowledge.txt"
CHECKPOINT  = "tiny_gpt.pt"

TEMP        = 0.7  # temperature

# Stop generation when the model starts a new question — keeps answers clean
STOP_SEQ    = "\nQ:"


# ── Device selection ──────────────────────────────────────────────────────────
def resolve_device(requested: Optional[str]) -> torch.device:
    """
    Resolve the compute device.

    Priority when --device is not passed:
      1. CUDA (cuda:0) if available
      2. MPS  (Apple Silicon) if available
      3. CPU  as fallback
    """
    if requested:
        dev = torch.device(requested)
        if dev.type == "cuda" and not torch.cuda.is_available():
            print("  WARNING: CUDA requested but not available — falling back to CPU.")
            return torch.device("cpu")
        if dev.type == "mps" and not torch.backends.mps.is_available():
            print("  WARNING: MPS requested but not available — falling back to CPU.")
            return torch.device("cpu")
        return dev

    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_info(device: torch.device) -> str:
    """Return a human-readable description of the selected device."""
    if device.type == "cuda":
        idx  = device.index if device.index is not None else 0
        name = torch.cuda.get_device_name(idx)
        mem  = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        return f"cuda:{idx}  ({name},  {mem:.1f} GB VRAM)"
    if device.type == "mps":
        return "mps  (Apple Silicon GPU)"
    return "cpu"

# ── Character-level tokeniser ─────────────────────────────────────────────────
def build_vocab(text):
    chars = sorted(set(text))
    stoi  = {c: i for i, c in enumerate(chars)}
    itos  = {i: c for c, i in stoi.items()}
    return stoi, itos

def encode(text, stoi):
    return [stoi[c] for c in text if c in stoi]

def decode(ids, itos):
    return "".join(itos.get(i, "?") for i in ids)

# ── Dataset ───────────────────────────────────────────────────────────────────
class CharDataset(torch.utils.data.Dataset):
    def __init__(self, data, ctx):
        self.data, self.ctx = data, ctx

    def __len__(self):
        return len(self.data) - self.ctx

    def __getitem__(self, i):
        x = self.data[i     : i + self.ctx]
        y = self.data[i + 1 : i + self.ctx + 1]
        return x, y

# ── Causal Self-Attention ─────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, ctx):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads

        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model,     bias=False)
        self.drop = nn.Dropout(DROPOUT)

        mask = torch.tril(torch.ones(ctx, ctx))
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)

        def to_heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        scale = math.sqrt(self.head_dim)
        att   = (q @ k.transpose(-2, -1)) / scale
        att   = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        att   = F.softmax(att, dim=-1)
        att   = self.drop(att)

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

# ── Transformer Block ─────────────────────────────────────────────────────────
class Block(nn.Module):
    def __init__(self, d_model, n_heads, ctx):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, ctx)
        self.ln2  = nn.LayerNorm(d_model)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, FFN_DIM),
            nn.GELU(),
            nn.Linear(FFN_DIM, d_model),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# ── Tiny GPT ──────────────────────────────────────────────────────────────────
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, ctx):
        super().__init__()
        self.ctx     = ctx
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(ctx, d_model)
        self.drop    = nn.Dropout(DROPOUT)
        self.blocks  = nn.Sequential(*[Block(d_model, n_heads, ctx) for _ in range(n_layers)])
        self.ln_f    = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: shares weights between embedding and output projection
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T   = idx.shape
        device = idx.device
        pos    = torch.arange(T, device=device)

        x      = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x      = self.blocks(x)
        x      = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, itos, stop_seq=STOP_SEQ,
                 max_new_tokens=300, temperature=0.7, top_k=20):
        """
        Auto-regressively sample one token at a time.
        Stops early if the generated text contains stop_seq (e.g. '\\nQ:'),
        so the model returns exactly one answer without running into the next question.
        """
        self.eval()
        generated = []

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.ctx:]
            logits, _ = self(idx_cond)
            logits    = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat([idx, next_tok], dim=1)

            generated.append(next_tok.item())

            # Stop as soon as the stop sequence appears in the tail of output
            tail = decode(generated[-len(stop_seq) - 2:], itos)
            if stop_seq in tail:
                break

        return idx

# ── Training ──────────────────────────────────────────────────────────────────
def train(device: torch.device, plotting=True):
    if not os.path.exists(TRAIN_FILE):
        print(f"ERROR: '{TRAIN_FILE}' not found.")
        sys.exit(1)

    text  = open(TRAIN_FILE, encoding="utf-8").read()
    stoi, itos = build_vocab(text)
    V     = len(stoi)
    data  = torch.tensor(encode(text, stoi), dtype=torch.long)
    ds    = CharDataset(data, CONTEXT)
    dl    = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True)

    model = TinyGPT(V, D_MODEL, N_HEADS, N_LAYERS, CONTEXT).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    print("=" * 56)
    print(f"  Device      : {device_info(device)}")
    print(f"  Vocabulary  : {V} unique characters")
    print(f"  Parameters  : {n_params:,}")
    print(f"  Training on : {len(data):,} tokens  |  {len(ds):,} samples")
    print(f"  Epochs      : {EPOCHS}")
    print("=" * 56)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    losses = []
    epochs = range(1, EPOCHS + 1)
    
    for epoch in epochs:
        model.train()
        total_loss = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg = total_loss / len(dl)
        losses.append(avg)

        if epoch % 100 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{EPOCHS}  |  loss {avg:.4f}  |  lr {scheduler.get_last_lr()[0]:.5f}")

    torch.save({
        "model":  {k: v.cpu() for k, v in model.state_dict().items()},
        "stoi":   stoi,
        "itos":   itos,
        "config": dict(d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, ctx=CONTEXT),
    }, CHECKPOINT)
    print(f"\n  Saved checkpoint → '{CHECKPOINT}'")
    
    if plotting:
        fig = plt.figure(figsize=(8,5))
        plt.plot(epochs, losses, lw=2, c='b')
        plt.grid(which='both', axis='both')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
        

# ── Inference ─────────────────────────────────────────────────────────────────
def ask(question: str, device: torch.device):
    if not os.path.exists(CHECKPOINT):
        print("No checkpoint found. Run:  python tiny_gpt.py train")
        sys.exit(1)

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    stoi, itos = ckpt["stoi"], ckpt["itos"]
    cfg  = ckpt.get("config", dict(d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, ctx=CONTEXT))
    V    = len(stoi)

    model = TinyGPT(V, cfg["d_model"], cfg["n_heads"], cfg["n_layers"], cfg["ctx"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"  Device: {device_info(device)}\n")

    prompt = f"Q: {question}\nA:"
    prompt = "".join(c for c in prompt if c in stoi)
    idx    = torch.tensor([encode(prompt, stoi)], dtype=torch.long).to(device)

    out    = model.generate(idx, temperature=TEMP, itos=itos, stop_seq=STOP_SEQ)

    # Decode and strip everything from the next "\nQ:" onward
    full   = decode(out[0].tolist(), itos)
    answer = full.split(STOP_SEQ)[0].strip()
    print(answer)

# # ── CLI ───────────────────────────────────────────────────────────────────────
# def main():
#     parser = argparse.ArgumentParser(
#         description="Tiny GPT — minimal transformer trained on Q&A pairs",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog=__doc__,
#     )
#     sub = parser.add_subparsers(dest="cmd")

#     p_train = sub.add_parser("train", help="Train on knowledge.txt")
#     p_train.add_argument(
#         "--device", type=str, default=None,
#         help="Compute device: cpu | cuda:0 | cuda:1 | mps  (default: auto-select best)",
#     )

#     p_ask = sub.add_parser("ask", help="Query the trained model")
#     p_ask.add_argument("question", nargs="+", help="Question string")
#     p_ask.add_argument(
#         "--device", type=str, default=None,
#         help="Compute device: cpu | cuda:0 | cuda:1 | mps  (default: auto-select best)",
#     )

#     args = parser.parse_args()

#     if args.cmd is None:
#         parser.print_help()
#         sys.exit(0)

#     device = resolve_device(args.device)

#     if args.cmd == "train":
#         train(device)
#     elif args.cmd == "ask":
#         ask(" ".join(args.question), device)


training_device = resolve_device('cuda:0')
inference_device = resolve_device('cpu')

start_time = time.time()
train(device=training_device)
end_time = time.time()
print(f"Training time: {(end_time - start_time) / 60.:.2f} mins.")

ask("Who invented C++?", device=inference_device)