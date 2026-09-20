import os
import torch

from config import DataConfig, ModelConfig, TrainConfig
from nika.dataset import TokenDataset
from nika.model import Nika
from nika.report import TrainReport
import matplotlib.pyplot as plt

@torch.no_grad() # ignore gradient since we are only measuring
def estimate_loss(model: Nika, ds: TokenDataset, eval_iters):
    model.eval() # switch to "measuring" mode
    
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = ds.get_batch()
        _, loss = model(x, y)
        losses[k] = loss.item()
    
    model.train() # back to "learning" mode
    return losses.mean().item()

if __name__ == "__main__":
    cfg, dcfg, mcfg = TrainConfig(), DataConfig(), ModelConfig()
    os.makedirs(os.path.dirname(cfg.checkpoint_path), exist_ok=True) # make checkpoint folder

    train_ds = TokenDataset(dcfg.train_bin_path, cfg.block_size, cfg.batch_size, cfg.device)
    val_ds = TokenDataset(dcfg.val_bin_path, cfg.block_size, cfg.batch_size, cfg.device)

    model = Nika(mcfg).to(cfg.device)
    print("params:", sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    best_val = float("inf") # best val loss
    history = [] # losses history
    report = TrainReport(cfg.checkpoint_path.replace(".pt", "_report.md"), model, cfg, mcfg, dcfg)
    
    plt.ion() # interactive: dont block on draw
    fig, (ax_tr, ax_va) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    train_line, = ax_tr.plot([], [], color="tab:blue")
    val_line, = ax_va.plot([], [], color="tab:orange")
    ax_tr.set_title("train loss"); ax_tr.set_xlabel("step"); ax_tr.set_ylabel("loss")
    ax_va.set_title("val loss"); ax_va.set_xlabel("step")
    fig.tight_layout()

    for step in range(cfg.max_iters):
        if step % cfg.eval_interval == 0: # validate
            tr = estimate_loss(model, train_ds, cfg.eval_iters)
            va = estimate_loss(model, val_ds, cfg.eval_iters)

            history.append((step, tr, va))
            report.update(step, tr, va)
            
            if va < best_val:
                best_val = va
                torch.save({"model": model.state_dict(), "cfg": mcfg, "history": history}, cfg.checkpoint_path)
                print(f"\nnew best saved\n")
            
            print(f"\nstep {step}: train {tr:.3f} | val {va:.3f}\n")
            
            steps = [h[0] for h in history]
            train_line.set_data(steps, [h[1] for h in history])
            val_line.set_data(steps, [h[2] for h in history])
            for ax in (ax_tr, ax_va):
                ax.relim(); ax.autoscale_view()
            plt.pause(0.001)

        x, y = train_ds.get_batch()
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    tr = estimate_loss(model, train_ds, cfg.eval_iters)
    va = estimate_loss(model, val_ds, cfg.eval_iters)
    history.append((cfg.max_iters, tr, va))
    report.update(cfg.max_iters, tr, va)
    if va < best_val: # the last measurement can be the best one too
        best_val = va
        torch.save({"model": model.state_dict(), "cfg": mcfg, "history": history}, cfg.checkpoint_path)
    print(f"\nfinal: train {tr:.3f} | val {va:.3f} | best val {best_val:.3f}\n")

    steps = [h[0] for h in history]
    train_line.set_data(steps, [h[1] for h in history])
    val_line.set_data(steps, [h[2] for h in history])
    for ax in (ax_tr, ax_va):
        ax.relim(); ax.autoscale_view()

    last_path = cfg.checkpoint_path.replace(".pt", "_last.pt")
    torch.save({"model": model.state_dict(), "cfg": mcfg, "history": history}, last_path)
    print("saved", cfg.checkpoint_path, "and", last_path)

    fig.savefig(cfg.checkpoint_path.replace(".pt", "_loss.png"))
    report.finish()
    plt.ioff()
    plt.show()