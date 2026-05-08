import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
import math
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    TOKEN_TRAIN_DIR, TOKEN_VAL_DIR,
    MODEL_CKPT_DIR,
    TRANSFORMER, TRANSFORMER_TRAIN,
)
from src.models.dataset        import MidiTokenDataset, collate_fn, VOCAB_SIZE, PAD_TOKEN
from src.models.Transformer_ae import MusicTransformer

def parse_args():
    p = argparse.ArgumentParser(description="Train MusicTransformer (fast mode)")
    p.add_argument("--data_root",      type=str,   default=str(Path(__file__).resolve().parents[2] / "data/preprocessed_output/tokens"))
    
    p.add_argument("--d_model",        type=int,   default=128)
    p.add_argument("--nhead",          type=int,   default=4)
    p.add_argument("--num_layers",     type=int,   default=3)
    p.add_argument("--dim_ff",         type=int,   default=512)
    p.add_argument("--max_len",        type=int,   default=2048)
    p.add_argument("--dropout",        type=float, default=0.1)
    # Training speed
    p.add_argument("--epochs",         type=int,   default=5)   # was 10
    p.add_argument("--batch_size",     type=int,   default=64)   # was 16
    p.add_argument("--seq_len",        type=int,   default=256)  # was 512
    p.add_argument("--stride",         type=int,   default=512)  # was 256 — bigger = fewer windows
    p.add_argument("--lr",             type=float, default=5e-4)
    p.add_argument("--warmup",         type=int,   default=200)
    p.add_argument("--clip_grad",      type=float, default=1.0)
    p.add_argument("--augment",        action="store_true", default=False)
    p.add_argument("--num_workers",    type=int,   default=2)
    p.add_argument("--log_every",      type=int,   default=200)  # was 50
    p.add_argument("--save_every",     type=int,   default=2)
    p.add_argument("--checkpoint_dir", type=str,   default=str(MODEL_CKPT_DIR))
    p.add_argument("--resume",         type=str,   default="")
    p.add_argument("--no_compile",     action="store_true",
                   help="Disable torch.compile (use if compile causes errors)")
    return p.parse_args()
 
 
def lr_lambda(step, warmup, total_steps):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
 
 
def train_epoch(model, loader, optimizer, scheduler, criterion,
                device, clip, scaler, log_every):
    model.train()
    total_loss, total_tokens = 0.0, 0
    t0 = time.time()
 
    for i, (src, tgt, pad_mask) in enumerate(loader):
        src, tgt, pad_mask = src.to(device), tgt.to(device), pad_mask.to(device)
        optimizer.zero_grad()
 
        if scaler is not None:  # AMP on CUDA
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(src, pad_mask)
                loss   = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(src, pad_mask)
            loss   = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
 
        scheduler.step()
        n = (tgt != PAD_TOKEN).sum().item()
        total_loss   += loss.item() * n
        total_tokens += n
 
        if (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            print(f"    step {i+1:>5}/{len(loader)}  "
                  f"loss={total_loss/max(1,total_tokens):.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"{elapsed:.0f}s elapsed")
 
    return total_loss / max(1, total_tokens)
 
 
@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for src, tgt, pad_mask in loader:
        src, tgt, pad_mask = src.to(device), tgt.to(device), pad_mask.to(device)
        with torch.autocast(device_type=device.type,
                            dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            logits = model(src, pad_mask)
            loss   = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        n = (tgt != PAD_TOKEN).sum().item()
        total_loss   += loss.item() * n
        total_tokens += n
    return total_loss / max(1, total_tokens)
 
 
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device      = {device}")
 
    root = Path(args.data_root)
 
    def collect(split):
        d = root / split
        if not d.exists():
            print(f"[train] WARNING: {d} not found — skipping")
            return []
        files = sorted(d.glob("*.json"))
        print(f"[train] {split:>12}: {len(files)} JSON files")
        return files
 
    train_files = collect("train")
    val_files   = collect("validation")
    if not train_files:
        raise FileNotFoundError(f"No JSON files in {root / 'train'}")
 
    train_ds = MidiTokenDataset(train_files, seq_len=args.seq_len,
                                stride=args.stride, augment=args.augment)
    val_ds   = MidiTokenDataset(val_files, seq_len=args.seq_len,
                                stride=args.stride) if val_files else None
 
    print(f"[train] train windows : {len(train_ds)}")
    if val_ds:
        print(f"[train] val   windows : {len(val_ds)}")
 
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    ) if val_ds else None
 
    # ── Smaller, faster model ──────────────────────────────────────────────────
    model = MusicTransformer(
        vocab_size  = VOCAB_SIZE,
        d_model     = args.d_model,
        nhead       = args.nhead,
        num_layers  = args.num_layers,
        dim_ff      = args.dim_ff,
        max_len     = args.max_len,
        dropout     = args.dropout,
    ).to(device)
 
    # torch.compile — free speedup on PyTorch >= 2.0
    if not args.no_compile and hasattr(torch, "compile"):
        print("[train] compiling model with torch.compile ...")
        model = torch.compile(model)
 
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] parameters    : {n_params:,}")
 
    optimizer   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    total_steps = args.epochs * len(train_loader)
    scheduler   = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: lr_lambda(s, args.warmup, total_steps))
    criterion   = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
 
    # AMP scaler — only on CUDA
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
 
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[train] resumed from epoch {ckpt['epoch']}")
 
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history = []
 
    steps_per_epoch = len(train_loader)
    print(f"[train] steps/epoch   : {steps_per_epoch}  "
          f"(batch={args.batch_size}, windows={len(train_ds)})")
    print(f"[train] estimated time: ~{steps_per_epoch * 0.05 / 60:.1f} min/epoch on CPU\n")
 
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"── epoch {epoch}/{args.epochs} ──────────────────")
        t0         = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 criterion, device, args.clip_grad,
                                 scaler, args.log_every)
        val_loss   = eval_epoch(model, val_loader, criterion, device) \
                     if val_loader else float("nan")
        elapsed    = time.time() - t0
        print(f"  ✓ train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"time={elapsed:.1f}s  ({elapsed/60:.1f}min)")
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "time_s": round(elapsed, 1)})
 
        if epoch % args.save_every == 0 or epoch == args.epochs:
            path = ckpt_dir / f"checkpoint_epoch{epoch:03d}.pt"
            # unwrap compiled model if needed
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save({
                "epoch":     epoch,
                "model":     raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args":      vars(args),
            }, path)
            print(f"  saved → {path}")
 
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(raw_model.state_dict(), ckpt_dir / "model_final.pt")
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("\n[train] done.")
 
 
if __name__ == "__main__":
    main()