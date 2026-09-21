"""Synthetic arithmetic in the A:/B: chat format.

Digits are written with spaces ("4 5") because the BPE tokenizer chunks numbers
inconsistently (' 23' is one token, ' 45' is two), and digit-aligned arithmetic
is impossible to learn from inconsistent chunks.

Two holdouts, testing different things:
  random 5% of pairs -> can it handle unseen COMBINATIONS of familiar numbers?
  every pair summing to HELDOUT_SUM -> can it produce an answer it has never emitted?
A model that memorised scores near zero on the second one.
"""
import argparse
import os
import random

from nika.tokenizer import EOT

MAX = 100
HELDOUT_SUM = 137
SEPARATOR = "\n" + EOT + "\n"

# Uniform sampling gives a zero operand in only ~1% of pairs, and that is where
# the first model failed: 26+0 -> 36, 6+0 -> 16, inventing a carry. Draw the
# edge cases deliberately instead.
P_ZERO = 0.08   # one operand is 0
P_SMALL = 0.10  # one operand is a single digit

TEMPLATES = [
    "what is {a} + {b}?",
    "{a} + {b} = ?",
    "add {a} and {b}",
    "what's the sum of {a} and {b}?",
    "calculate {a} + {b}",
]


def spaced(n, reverse=False):
    """45 -> '4 5', one token per digit; reversed -> '5 4'

    Answers are written least significant digit first. Addition carries right to
    left, but the model generates left to right, so in normal order it has to emit
    the leading digit before it has computed the carries that decide it. Reversed,
    every digit only depends on columns it has already produced.
    """
    s = str(n)[::-1] if reverse else str(n)
    return " ".join(s)


def sample_pair():
    """mostly uniform, with zero and single-digit operands oversampled"""
    r = random.random()
    if r < P_ZERO:
        a, b = random.randrange(MAX), 0
    elif r < P_ZERO + P_SMALL:
        a, b = random.randrange(MAX), random.randrange(10)
    else:
        a, b = random.randrange(MAX), random.randrange(MAX)

    if random.random() < 0.5: # the special operand should appear on either side
        a, b = b, a
    return a, b


def is_heldout(a, b):
    # deterministic, unlike hash(), which is salted per process
    return a + b == HELDOUT_SUM or (a * MAX + b) % 20 == 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50_000)
    p.add_argument("--out", default="data/tasks.txt")
    p.add_argument("--heldout", default="data/tasks_heldout.txt")
    args = p.parse_args()

    random.seed(0) # same data every run
    os.makedirs("data", exist_ok=True)

    n_train = 0
    heldout = {} # question -> answer, deduped so each pair is scored once

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for _ in range(args.n):
            a, b = sample_pair()
            question = random.choice(TEMPLATES).format(a=spaced(a), b=spaced(b))
            answer = spaced(a + b, reverse=True) # least significant digit first
            example = f"A: {question}\nB: {answer}"

            if is_heldout(a, b):
                heldout[question] = answer
            else:
                f.write(example + SEPARATOR)
                n_train += 1

    with open(args.heldout, "w", encoding="utf-8", newline="\n") as f:
        for question, answer in heldout.items():
            f.write(f"A: {question}\nB: {answer}" + SEPARATOR)

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"{n_train} training examples ({size_mb:.1f} MB) -> {args.out}")
    print(f"{len(heldout)} unique held-out questions -> {args.heldout}")
    print(f"  of which sum to {HELDOUT_SUM}: "
          f"{sum(1 for q in heldout if heldout[q] == spaced(HELDOUT_SUM, reverse=True))}")
