import argparse
import os
import shutil
from dataclasses import replace

import torch

from config import DataConfig, ModelConfig, TrainConfig
from nika.dataset import TokenDataset
from nika.model import Nika
from nika.report import TrainReport
from nika.schedule import get_lr

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args(cfg, mcfg, dcfg):
    """command line overrides, so Colab can change knobs without editing config.py"""
    p = argparse.ArgumentParser()
    p.add_argument("--n-embed", type=int, default=mcfg.n_embed)
    p.add_argument("--n-layer", type=int, default=mcfg.n_layer)
    p.add_argument("--n-head", type=int, default=mcfg.n_head)
    p.add_argument("--block-size", type=int, default=cfg.block_size) # both configs
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--max-iters", type=int, default=cfg.max_iters)
    p.add_argument("--lr", type=float, default=cfg.learning_rate)
    p.add_argument("--eval-interval", type=int, default=cfg.eval_interval)
    p.add_argument("--eval-iters", type=int, default=cfg.eval_iters)
    p.add_argument("--warmup-iters", type=int, default=cfg.warmup_iters)
    p.add_argument("--device", default=cfg.device)
    p.add_argument("--checkpoint", default=cfg.checkpoint_path)
    p.add_argument("--train-bin", default=dcfg.train_bin_path)
    p.add_argument("--val-bin", default=dcfg.val_bin_path)
    p.add_argument("--resume", action="store_true", help="continue from the _last.pt checkpoint")
    # checkpoints are written locally every eval (fast) and copied to a slow, permanent
    # place (Google Drive) every --backup-every evals
    p.add_argument("--backup-dir", default="", help="copy checkpoints here, e.g. a Drive folder")
    p.add_argument("--backup-every", type=int, default=5, help="evals between backups")
    a = p.parse_args()

    cfg = replace(cfg, block_size=a.block_size, batch_size=a.batch_size, max_iters=a.max_iters,
                  learning_rate=a.lr, eval_interval=a.eval_interval, eval_iters=a.eval_iters,
                  warmup_iters=a.warmup_iters, device=a.device, checkpoint_path=a.checkpoint)
    mcfg = replace(mcfg, n_embed=a.n_embed, n_layer=a.n_layer, n_head=a.n_head, block_size=a.block_size)
    dcfg = replace(dcfg, train_bin_path=a.train_bin, val_bin_path=a.val_bin)
    return cfg, mcfg, dcfg, a


def pick_amp(device):
    """bf16 where supported (Ampere+), else fp16 + a gradient scaler (Colab's T4)"""
    if device != "cuda":
        return torch.float32, False
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, False
    return torch.float16, True


def save(path, **payload):
    """write to a temp file first: a crash mid-save leaves the old checkpoint intact"""
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def backup(paths, backup_dir):
    """copy finished files to a permanent place; skips whatever does not exist yet"""
    os.makedirs(backup_dir, exist_ok=True)
    for path in paths:
        if os.path.exists(path):
            shutil.copy2(path, os.path.join(backup_dir, os.path.basename(path)))
    print(f"backed up to {backup_dir}")


def restore(paths, backup_dir):
    """pull checkpoints back from the backup when the local disk is empty (new session)"""
    for path in paths:
        src = os.path.join(backup_dir, os.path.basename(path))
        if os.path.exists(src) and not os.path.exists(path):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            shutil.copy2(src, path)
            print(f"restored {path} from backup")


@torch.no_grad() # ignore gradient since we are only measuring
def estimate_loss(model: Nika, ds: TokenDataset, eval_iters, device, amp_dtype):
    model.eval() # switch to "measuring" mode

    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = ds.get_batch()
        with torch.autocast(device, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            _, loss = model(x, y)
        losses[k] = loss.item()

    model.train() # back to "learning" mode
    return losses.mean().item()


if __name__ == "__main__":
    cfg, mcfg, dcfg, args = parse_args(TrainConfig(), ModelConfig(), DataConfig())
    os.makedirs(os.path.dirname(cfg.checkpoint_path), exist_ok=True) # make checkpoint folder
    last_path = cfg.checkpoint_path.replace(".pt", "_last.pt")
    report_path = cfg.checkpoint_path.replace(".pt", "_report.md")
    plot_path = cfg.checkpoint_path.replace(".pt", "_loss.png")
    backup_paths = [cfg.checkpoint_path, last_path, report_path, plot_path]

    amp_dtype, need_scaler = pick_amp(cfg.device)
    scaler = torch.amp.GradScaler(cfg.device, enabled=need_scaler)
    print(f"device {cfg.device} | autocast {amp_dtype} | grad scaler {need_scaler}")

    train_ds = TokenDataset(dcfg.train_bin_path, cfg.block_size, cfg.batch_size, cfg.device)
    val_ds = TokenDataset(dcfg.val_bin_path, cfg.block_size, cfg.batch_size, cfg.device)

    start_step, best_val, history = 0, float("inf"), []
    if args.resume and args.backup_dir:
        restore(backup_paths, args.backup_dir) # a new Colab session starts with an empty disk
    if args.resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=cfg.device, weights_only=False)
        mcfg = ckpt["cfg"] # the checkpoint's architecture wins, or the weights would not fit
        model = Nika(mcfg).to(cfg.device)
        model.load_state_dict(ckpt["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
        optimizer.load_state_dict(ckpt["optimizer"]) # adam's momentum, or the loss jumps on resume
        scaler.load_state_dict(ckpt["scaler"])
        start_step, best_val, history = ckpt["step"], ckpt["best_val"], ckpt["history"]
        print(f"resumed from {last_path} at step {start_step}, best val {best_val:.3f}")
    else:
        model = Nika(mcfg).to(cfg.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    print("params:", sum(p.numel() for p in model.parameters()))
    report = TrainReport(report_path, model, cfg, mcfg, dcfg, rows=history)

    def checkpoint(step, val_loss):
        state = dict(model=model.state_dict(), cfg=mcfg, history=history,
                     optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                     step=step, best_val=best_val)
        save(last_path, **state) # for resuming
        if val_loss <= best_val:
            save(cfg.checkpoint_path, **state) # best so far, what generate.py loads
            print("new best saved")

    n_evals = 0
    for step in range(start_step, cfg.max_iters):
        if step % cfg.eval_interval == 0: # validate
            tr = estimate_loss(model, train_ds, cfg.eval_iters, cfg.device, amp_dtype)
            va = estimate_loss(model, val_ds, cfg.eval_iters, cfg.device, amp_dtype)

            history.append((step, tr, va))
            report.update(step, tr, va)
            best_val = min(best_val, va)
            checkpoint(step, va)
            print(f"step {step}: train {tr:.3f} | val {va:.3f}")

            n_evals += 1
            if args.backup_dir and n_evals % args.backup_every == 0:
                backup(backup_paths, args.backup_dir)

        x, y = train_ds.get_batch()
        with torch.autocast(cfg.device, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward() # a no-op multiply by 1.0 when the scaler is disabled

        lr = get_lr(step, cfg.max_iters, cfg.learning_rate, cfg.warmup_iters, cfg.min_lr_frac)
        for group in optimizer.param_groups:
            group["lr"] = lr
        scaler.step(optimizer)
        scaler.update()

    tr = estimate_loss(model, train_ds, cfg.eval_iters, cfg.device, amp_dtype)
    va = estimate_loss(model, val_ds, cfg.eval_iters, cfg.device, amp_dtype)
    history.append((cfg.max_iters, tr, va))
    report.update(cfg.max_iters, tr, va)
    best_val = min(best_val, va)
    checkpoint(cfg.max_iters, va)
    print(f"\nfinal: train {tr:.3f} | val {va:.3f} | best val {best_val:.3f}")
    print("saved", cfg.checkpoint_path, "and", last_path)

    report.finish()
    if args.backup_dir:
        backup(backup_paths, args.backup_dir)
