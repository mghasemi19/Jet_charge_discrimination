import sys, os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from tabulate import tabulate
import h5py
import awkward as ak
import math

read_file = True
if read_file:
    final_df = pd.read_csv('./uubar.csv')
    print(tabulate(final_df.head(), headers='keys', tablefmt='psql'))
    print(tabulate(final_df.tail(), headers='keys', tablefmt='psql'))
print(final_df.shape)

NDIM = len(final_df.keys()) - 1
#dataset = final_df.values

# Count NaNs in each column
df_nonan = final_df.copy()
df_nonan = df_nonan.dropna()
#print(df_nonan.isna().sum())
dataset = df_nonan.values
X = dataset[:,0:NDIM]
Y = dataset[:,NDIM]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7)

# preprocessing: standard scalar
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Check for NaNs/Infs in the dataset
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

# GPU-ready training template for the PlainParticleTransformer (no interaction embedding)
# - Moves model + batches to CUDA (if available)
# - Verbose logs: batch loss (optional), epoch loss, accuracy, AUC
# - Uses AMP (mixed precision) for speed on modern GPUs
#
# Assumes already have X (N, 69) and y (N,) in numpy arrays or torch tensors.

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Sequence

# ----------------------------
# 1) Transformer model includes ParticleEmbed, ParticleBlock, ClassAttentionBlock, and PlainParticleTransformer  
# ----------------------------

# Plain (no-interaction) Particle Transformer for tabular / constituent-like inputs
# - 1. Particle embedding (MLP: 128 -> 512 -> 128 with LayerNorm + GELU)
# - 2. Particle attention blocks (standard MHSA, no pair/interaction bias)
# - 3. Class attention blocks (CaiT-style: cls queries particles)
#
# Works with:
#   x_particles: (B, P, F)  where:
#       B = batch size
#       P = number of “particles/tokens” per event (can be 69 if you treat each feature as a token)
#       or (B, 69, 1) each feature is treated as a “particle”
#       F = features per particle/token
#   mask (optional): (B, P) with 1 for real tokens and 0 for padded tokens


class ParticleEmbed(nn.Module):
    """
    Per-particle MLP embedding like ParT:
      Linear -> LN -> GELU  (x3) with widths [128, 512, embed_dim]
    """
    def __init__(self, input_dim: int, embed_dims: Sequence[int] = (128, 512, 128)):
        super().__init__()
        dims = [input_dim] + list(embed_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.GELU(),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, P, F)  like (B, 69, 1) each feature is treated as a “particle” -- token dim = 1
        return self.net(x)  # (B, P, C)


class ParticleBlock(nn.Module):
    """
    Transformer block (pre-LN) for particle self-attention.
    No interaction embedding / no pair bias.
    PyTorch module (a reusable layer). It will contain:
    self-attention
    feed-forward network (FFN)
    residual connections
    normalization
    dropout
    """
    def __init__(
        self,
        embed_dim: int = 128,                # vector size for each token (C)
        num_heads: int = 8,                  # number of attention heads               
        ffn_ratio: int = 4,                  # hidden size in FFN = embed_dim * ffn_ratio
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        activation_dropout: float = 0.1,
    ):
        super().__init__()

        # Multi-head self-attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,  # (B, P, C)
        )
        self.drop1 = nn.Dropout(dropout)

        # Feed-forward network (MLP)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = embed_dim * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(activation_dropout),
            nn.Linear(hidden, embed_dim),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, P, C)
        # key_padding_mask: (B, P) with True for PAD positions (PyTorch convention)
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)    # h for query, key, value
        x = x + self.drop1(h)

        h = self.norm2(x)
        h = self.ffn(h)
        x = x + self.drop2(h)
        return x


class ClassAttentionBlock(nn.Module):
    """
    CaiT-style class attention:
      - cls token attends to [cls + particles]
      - particles are NOT updated in cls blocks (only cls updates)
    """
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        ffn_ratio: int = 4,
        dropout: float = 0.0,        # ParT often uses 0 in cls blocks
        attn_dropout: float = 0.0,
        activation_dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,  # (B, T, C)
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = embed_dim * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(activation_dropout),
            nn.Linear(hidden, embed_dim),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x_particles: torch.Tensor,          # (B, P, C)
        x_cls: torch.Tensor,                # (B, 1, C)
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Build tokens [cls, particles]
        u = torch.cat([x_cls, x_particles], dim=1)  # (B, 1+P, C)

        # Expand padding mask to include cls (cls is never padded)
        if key_padding_mask is not None:
            cls_pad = torch.zeros((key_padding_mask.size(0), 1), device=key_padding_mask.device, dtype=torch.bool)
            pad_u = torch.cat([cls_pad, key_padding_mask], dim=1)  # (B, 1+P)
        else:
            pad_u = None

        # Attention: query = cls only, key/value = [cls + particles]
        residual = x_cls
        q = self.norm1(x_cls)  # (B, 1, C)
        kv = self.norm1(u)     # (B, 1+P, C)
        h, _ = self.attn(q, kv, kv, key_padding_mask=pad_u, need_weights=False)
        x_cls = residual + self.drop1(h)

        # FFN on cls only
        residual = x_cls
        h = self.norm2(x_cls)
        h = self.ffn(h)
        x_cls = residual + self.drop2(h)
        return x_cls  # (B, 1, C)


class PlainParticleTransformer(nn.Module):
    """
    Full model:
      embed -> N particle blocks -> M class-attention blocks -> norm -> head
    """
    def __init__(
        self,
        input_dim: int,                     # features per token/particle
        num_classes: int = 2,
        num_tokens: Optional[int] = None,   # not required; kept for clarity
        embed_dims=(128, 512, 128),         # embedding dimensions for ParticleEmbed MLP
        embed_dim: int = 128,               # must match last of embed_dims
        num_heads: int = 8,
        num_layers: int = 8,
        num_cls_layers: int = 2,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        cls_dropout: float = 0.0,           # typical ParT default for cls blocks
        fc_hidden: Optional[int] = None,    # e.g., 256 if you want an extra FC layer
    ):
        super().__init__()

        assert embed_dim == embed_dims[-1], "embed_dim must equal embed_dims[-1]"

        self.embed = ParticleEmbed(input_dim=input_dim, embed_dims=embed_dims)

        self.blocks = nn.ModuleList([
            ParticleBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
                activation_dropout=activation_dropout,
            )
            for _ in range(num_layers)
        ])

        self.cls_blocks = nn.ModuleList([
            ClassAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                dropout=cls_dropout,
                attn_dropout=cls_dropout,
                activation_dropout=cls_dropout,
            )
            for _ in range(num_cls_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        if fc_hidden is None:
            self.head = nn.Linear(embed_dim, num_classes)
        else:
            self.head = nn.Sequential(
                nn.Linear(embed_dim, fc_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_hidden, num_classes),
            )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, P, F)
        mask (optional): (B, P) with 1 for real tokens and 0 for padded tokens
        """
        if mask is None:
            key_padding_mask = None
        else:
            # PyTorch MHA expects True where positions are padded
            key_padding_mask = ~mask.bool()  # (B, P)

        # Embed particles
        x = self.embed(x)  # (B, P, C)

        # Zero-out padded tokens (optional but often helps)
        if mask is not None:
            x = x.masked_fill((~mask.bool()).unsqueeze(-1), 0.0)

        # Particle (self) attention blocks
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)

        # Class attention blocks (only update cls token)
        B = x.size(0)
        cls = self.cls_token.expand(B, 1, -1)  # (B, 1, C)
        for cblk in self.cls_blocks:
            cls = cblk(x_particles=x, x_cls=cls, key_padding_mask=key_padding_mask)

        cls = self.norm(cls).squeeze(1)  # (B, C)
        logits = self.head(cls)          # (B, num_classes)
        return logits
    
    # ----------------------------
# 2) Dataset + split (no sklearn)
# ----------------------------

class TabJets(Dataset):
    """
    If your data is (N, 69) flat features:
      we treat each scalar feature as a token -> (69, 1)
    """
    def __init__(self, X, y):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).long()
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        x = self.X[i].unsqueeze(-1)  # (69,) -> (69,1)
        y = self.y[i]
        return x, y

def simple_split_indices(n, val_frac=0.2, seed=123):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = int(n * val_frac)
    val_idx = perm[:n_val]
    trn_idx = perm[n_val:]
    return trn_idx, val_idx

def stratified_split_indices(y, val_frac=0.2, seed=123):
    """
    y: torch tensor (N,) or numpy array (N,) with class labels 0/1
    returns: train_idx, val_idx as torch.LongTensor
    """
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
    y = y.long()

    g = torch.Generator().manual_seed(seed)

    idx0 = torch.where(y == 0)[0]
    idx1 = torch.where(y == 1)[0]

    # shuffle within each class
    idx0 = idx0[torch.randperm(idx0.numel(), generator=g)]
    idx1 = idx1[torch.randperm(idx1.numel(), generator=g)]

    n0_val = max(1, int(idx0.numel() * val_frac))
    n1_val = max(1, int(idx1.numel() * val_frac))

    val_idx = torch.cat([idx0[:n0_val], idx1[:n1_val]])
    trn_idx = torch.cat([idx0[n0_val:], idx1[n1_val:]])

    # shuffle final indices
    val_idx = val_idx[torch.randperm(val_idx.numel(), generator=g)]
    trn_idx = trn_idx[torch.randperm(trn_idx.numel(), generator=g)]

    return trn_idx, val_idx

# ----------------------------
# 3) Metrics helpers
# ----------------------------

@torch.no_grad()
def accuracy_from_logits(logits, y):
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()

@torch.no_grad()
def auc_binary_from_probs(probs_pos, y_true):
    """
    Simple AUC without sklearn (O(N log N)).
    probs_pos: (N,) probabilities of class 1
    y_true: (N,) 0/1
    """
    # Sort by score descending
    scores, idx = torch.sort(probs_pos, descending=True)
    y = y_true[idx].float()
    # Compute ranks-based AUC
    # AUC = (sum ranks of positives - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
    # Here rank 1 is lowest; easier if we sort ascending:
    scores_a, idx_a = torch.sort(probs_pos, descending=False)
    y_a = y_true[idx_a].float()
    ranks = torch.arange(1, y_a.numel() + 1, device=y_a.device).float()
    n_pos = y_a.sum()
    n_neg = y_a.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = (ranks * y_a).sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc.item()

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_n = 0
    all_probs = []
    all_y = []
    ce = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = ce(logits, y)
        total_loss += loss.item()
        total_n += y.numel()
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.append(probs.detach())
        all_y.append(y.detach())
    all_probs = torch.cat(all_probs)
    all_y = torch.cat(all_y)
    avg_loss = total_loss / max(total_n, 1)
    #acc = accuracy_from_logits(torch.stack([1-all_probs, all_probs], dim=1), all_y)
    # compute acc from logits for correctness
    # easiest: store logits too, or recompute via threshold:
    pred = (all_probs >= 0.5).long()
    acc = (pred == all_y).float().mean().item()
    auc = auc_binary_from_probs(all_probs, all_y)
    return avg_loss, acc, auc

import torch
import numpy as np
import matplotlib.pyplot as plt

@torch.no_grad()
def collect_probs_labels(model, loader, device):
    model.eval()
    probs_all = []
    y_all = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1]  # P(class=1)
        probs_all.append(probs.detach().cpu())
        y_all.append(y.detach().cpu())
    probs_all = torch.cat(probs_all).numpy()
    y_all = torch.cat(y_all).numpy().astype(np.int64)
    return probs_all, y_all

# ----------------------------
# 4) Training loop with CUDA + verbose
# ----------------------------

def train_model(
    X, Y,
    epochs=1,
    batch_size=512,
    lr=2e-4,
    weight_decay=1e-4,
    val_frac=0.2,
    seed=123,
    log_every=50,          # print every N batches
    use_amp=True,
    num_workers=2,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | CUDA available: {torch.cuda.is_available()}")

    ds = TabJets(X, Y)
    #trn_idx, val_idx = simple_split_indices(len(ds), val_frac=val_frac, seed=seed)

    # stratified split to guarantee both classes exist in val
    trn_idx, val_idx = stratified_split_indices(ds.y, val_frac=val_frac, seed=seed)

    # sanity prints
    y_trn = ds.y[trn_idx]
    y_val = ds.y[val_idx]
    print("Train class counts:", torch.bincount(y_trn))
    print("Val   class counts:", torch.bincount(y_val))

    trn_ds = torch.utils.data.Subset(ds, trn_idx.tolist())
    val_ds = torch.utils.data.Subset(ds, val_idx.tolist())

    loader_trn = DataLoader(
        trn_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    loader_val = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    model = PlainParticleTransformer(
        input_dim=1,        # because tokens are scalars (68,1)
        num_classes=2,
        num_layers=8,
        num_cls_layers=2,
        num_heads=8,
        dropout=0.1,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    best_val_auc = -1.0
    history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
    "val_auc": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n_seen = 0

        for step, (x, yb) in enumerate(loader_trn, start=1):
            x = x.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                logits = model(x)
                loss = ce(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item() * yb.size(0)
            n_seen += yb.size(0)

            if log_every and (step % log_every == 0):
                # quick batch diagnostics
                with torch.no_grad():
                    acc_b = accuracy_from_logits(logits, yb)
                avg = running / max(n_seen, 1)
                print(f"Epoch {epoch:03d} | step {step:04d}/{len(loader_trn)} | "
                      f"loss {avg:.4f} | batch_acc {acc_b:.3f}")

        trn_loss = running / max(n_seen, 1)
        # Compute train accuracy at end of epoch
        model.eval()
        with torch.no_grad():
            all_probs_tr = []
            all_y_tr = []
            for x_tr, y_tr in loader_trn:
                x_tr = x_tr.to(device)
                y_tr = y_tr.to(device)
                logits_tr = model(x_tr)
                probs_tr = torch.softmax(logits_tr, dim=1)[:, 1]
                all_probs_tr.append(probs_tr)
                all_y_tr.append(y_tr)

            all_probs_tr = torch.cat(all_probs_tr)
            all_y_tr = torch.cat(all_y_tr)
            train_acc = ((all_probs_tr >= 0.5).long() == all_y_tr).float().mean().item()

        val_loss, val_acc, val_auc = evaluate(model, loader_val, device)
        dt = time.time() - t0

        improved = val_auc > best_val_auc
        if improved:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # Store metrics
        history["train_loss"].append(trn_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        print(f"[Epoch {epoch:03d}] time {dt:.1f}s | "
              f"train_loss {trn_loss:.4f} | val_loss {val_loss:.4f} | "
              f"val_acc {val_acc:.3f} | val_auc {val_auc:.4f} "
              f"{'<= BEST' if improved else ''}")

    # restore best
    model.load_state_dict(best_state)
    model.to(device)
    print(f"Best val AUC: {best_val_auc:.4f}")
    return model, device, history

# ----------------------------
# 5) Example call
# ----------------------------
def main():
    model, device, history = train_model(
        X, Y,
        epochs=100,
        batch_size=512,
        log_every=25,
        use_amp=True,
        num_workers=0,   # now you can use >0 safely
    )
    return model, device, history

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()   # important on Windows
    model, device, history = main()
