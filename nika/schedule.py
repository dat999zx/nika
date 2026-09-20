import math

def get_lr(step, max_iters, lr, warmup_iters=200, min_lr_frac=0.1):
    if step < warmup_iters:
        return lr * (step + 1) / warmup_iters

    progress = (step - warmup_iters) / max(1, max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr * (min_lr_frac + (1 - min_lr_frac) * coeff)

if __name__ == "__main__":
    for s in [0, 50, 100, 200, 1000, 2000, 4000, 6000, 8000]:
        print(s, round(get_lr(s, 8000, 1e-3), 6))