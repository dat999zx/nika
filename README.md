# Nika

A small GPT-style language model built from scratch in PyTorch, as a way to learn how transformers actually work. No Hugging Face model code: the tokenizer, attention, blocks and training loop are all written here.

It was then pushed as far as a free GPU allows: pretrained on ~590M tokens of web text, and fine-tuned to chat, answer questions about itself, recall things you told it, and do two-digit addition.

| stage | params | result |
| --- | --- | --- |
| bigram baseline | 0.5M | val loss 5.28 |
| 4 blocks, 256 wide (local) | 5.3M | val loss 3.93 |
| 6 blocks, 512 wide (Colab T4) | 23.2M | val loss 3.06 |
| + addition fine-tune | 23.2M | 98.9% on held-out sums, 100% on answers never seen in training |
| + chat fine-tune | 23.2M | identity works, in-context recall partly, general knowledge does not |

It is a learning project, not a useful assistant. See [What it can and cannot do](#what-it-can-and-cannot-do).

---

## Setup

Python 3.11+ and an NVIDIA GPU are assumed. Everything runs on CPU too, just slowly.

```
python -m venv .venv
.venv\Scripts\activate                 # Windows. On Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install numpy datasets matplotlib
```

**Every script runs as a module from the repo root**, because they import from `nika/` and `config.py`:

```
python -m scripts.train          # works
python scripts/train.py          # fails: cannot find `nika`
```

---

## Reading order

If you want to understand the model, read the code in this order. Each file only depends on the ones above it.

| # | file | what it teaches |
| --- | --- | --- |
| 1 | [`nika/tokenizer.py`](nika/tokenizer.py) | Byte-level BPE: how text becomes numbers. Count pairs, glue the most common, repeat. |
| 2 | [`nika/dataset.py`](nika/dataset.py) | Random windows from a token file, and the one-step shift that turns text into its own labels. |
| 3 | [`nika/attention.py`](nika/attention.py) | Q, K, V, the √d_k scaling, the causal mask. `AttentionHead` is the readable version; `FastMultiHeadAttention` does all heads in one batched matmul. The file's self-test proves no token can see the future. |
| 4 | [`nika/mlp.py`](nika/mlp.py) | The per-token network: widen 4×, GELU, shrink back. |
| 5 | [`nika/block.py`](nika/block.py) | One transformer block: pre-LayerNorm, attention, residual, MLP, residual. |
| 6 | [`nika/model.py`](nika/model.py) | Embeddings, positions, a stack of blocks, the LM head, the loss, and `generate`. |
| 7 | [`nika/schedule.py`](nika/schedule.py) | Warmup plus cosine learning-rate decay. |
| 8 | [`scripts/train.py`](scripts/train.py) | The training loop. The core is five lines (batch, forward, zero_grad, backward, step); the rest is mixed precision, checkpointing and resume. |
| 9 | [`scripts/generate.py`](scripts/generate.py), [`scripts/chat.py`](scripts/chat.py) | Sampling, temperature, top-k, and why a chatbot has to resend the whole conversation every turn. |

Everything else in `scripts/` is data plumbing and evaluation, and can be read as needed.

---

## Layout

```
config.py              every size and path, as dataclasses. Scripts override them from the command line.
nika/                  the library: tokenizer, model pieces, dataset, LR schedule, training report
scripts/               things you run: download, tokenize, train, fine-tune, generate, chat, evaluate
notebooks/             the Colab notebook for the 23M run
data/                  generated files, gitignored except tokenizer.txt (the vocabulary everything depends on)
checkpoint/            model weights and training reports, gitignored
```

---

## Running it

The pipeline has four stages. Each one writes files the next one reads.

```
text  ->  tokenizer  ->  token files  ->  pretrained base  ->  fine-tuned model
```

### 1. Data and tokenizer

```
python -m scripts.download_data --mb 100        # FineWeb-Edu -> data/tiny.txt
python -m scripts.train_tokenizer               # ~11 min. Skip it: data/tokenizer.txt is committed
python -m scripts.encode_data                   # -> data/train.bin, data/val.bin
```

`download_data` finishes correctly and may then print `Bad file descriptor ... Retrying`. That is the `datasets` library complaining about the abandoned stream; the file is complete and the process can be interrupted.

Do not retrain the tokenizer unless you mean to. Every checkpoint maps token ids to meanings, and a new tokenizer changes every id.

### 2. Pretraining

Locally, the small model (5.3M params, ~5 min on an RTX 3050):

```
python -m scripts.train
```

On Colab, the 23M model: open [`notebooks/nika_colab.ipynb`](notebooks/nika_colab.ipynb) and run the cells in order. It downloads 2 GB of text, caches the encoded data on Google Drive, trains in the background, and shows the report live. Or directly:

```
python -m scripts.train --n-embed 512 --n-layer 6 --n-head 8 --block-size 256 \
    --batch-size 48 --max-iters 48000 --lr 6e-4 --warmup-iters 500 \
    --checkpoint checkpoint/nika30m.pt --resume
```

Useful flags:

| flag | what it does |
| --- | --- |
| `--resume` | continue from `<checkpoint>_last.pt`: weights, optimizer, step and history |
| `--backup-dir DIR --backup-every N` | copy checkpoints to slow permanent storage (Drive) every N evals; `--resume` restores from there on a fresh machine |
| `--amp-dtype {auto,bf16,fp16,fp32}` | `auto` picks bf16 only on GPUs that do it natively, fp16 + a gradient scaler otherwise |
| `--init-from PATH` | copy weights from a checkpoint but start a fresh run: new optimizer, new schedule. This is fine-tuning. |

While training, `checkpoint/<name>_report.md` is rewritten at every eval with the config, the loss table and the curve. Open it in VS Code with `Ctrl+Shift+V` for a live view.

### 3. Sampling

```
python -m scripts.generate --checkpoint checkpoint/nika30m.pt --prompt "Water is" --temperature 0.8
```

### 4. Fine-tuning

Build the fine-tune data. Every generator writes `A:`/`B:` conversations separated by `<|EOT|>`:

```
python -m scripts.download_dialog               # DailyDialog -> data/dialog.txt
python -m scripts.gen_recall                    # "my name is X" ... "what is my name?"
python -m scripts.gen_persona                   # "who are you?" and friends
python -m scripts.gen_facts                     # a glossary from Simple English Wikipedia
python -m scripts.gen_tasks                     # two-digit addition, with held-out slices
```

Mix them document by document, encode, and fine-tune from the base:

```
python -m scripts.mix_data --inputs data/dialog.txt data/recall.txt data/persona.txt --out data/finetune.txt
python -m scripts.encode_data --data data/finetune.txt --train-bin data/ft_train.bin --val-bin data/ft_val.bin
python -m scripts.train --init-from checkpoint/nika30m.pt --train-bin data/ft_train.bin --val-bin data/ft_val.bin \
    --checkpoint checkpoint/nika_chat.pt --batch-size 16 --block-size 256 --lr 2e-4 --max-iters 4000 --warmup-iters 100
```

Always use `mix_data` rather than concatenating files. The encoder splits train/val contiguously, so concatenated sources put one source entirely in the validation set.

### 5. Talking to it and measuring it

```
python -m scripts.chat --checkpoint checkpoint/nika_chat_last.pt
python -m scripts.eval_tasks --checkpoint checkpoint/nika_math_last.pt --limit 400
```

`chat.py` spaces out the digits you type and un-reverses the model's arithmetic answers, so you can type `23 + 45` normally. `eval_tasks.py` scores exact-match accuracy on the held-out sums, with greedy decoding.

---

## What it can and cannot do

What worked was always either a **procedure** or a **small fixed set of facts**. What failed needed **stored knowledge**.

| capability | result | why |
| --- | --- | --- |
| two-digit addition | ✅ 98.9%, and 100% on sums never seen | a procedure, learnable from data |
| "who are you?" | ✅ when asked early in a conversation | a handful of fixed facts |
| "what is my name?" after telling it | 🟡 copies held-out cities, sometimes guesses names | copying from context is what attention does |
| "what is machine learning?" | ❌ | 3,700 definitions is too much to store in 23M params |
| answering arbitrary questions | ❌ | needs knowledge and instruction tuning at ~1B+ params |

---

## Things learned the hard way

These are the findings worth more than the code.

**Reversing the answer digits took addition from 5.5% to 95%.** Addition carries right to left, but the model writes left to right, so in normal order it must emit the leading digit before computing the carries that decide it. Written least-significant-digit first, every digit depends only on columns already produced.

**Numbers needed spaces between digits.** BPE split them inconsistently (`" 23"` is one token, `" 45"` is two), which makes column arithmetic impossible to learn. This is why real tokenizers treat digits specially.

**A bug in the data split looked exactly like a model weakness.** Every arithmetic error involved a zero operand. The cause was `(a * 100 + b) % 20`: since 100 is divisible by 20, every pair with `b = 0` landed in the test set and none in training. Oversampling zeros made the score worse. When errors cluster in one class, check the split first.

**Validation loss can measure the wrong thing.** In the arithmetic fine-tune, val loss was dominated by dialogue and said the model was getting worse while arithmetic accuracy kept improving. Pick checkpoints by the metric you care about.

**The model learns the situation, not just the answer.** Identity questions were answered perfectly as the first turn of a conversation and failed after a few turns of small talk, because every training example asked them in a conversation about nothing else.

**Mixing beats sequential fine-tuning.** Fine-tuning on one skill after another makes the model forget the earlier ones. Interleaving the data in every batch kept both chat and arithmetic.

**On a T4, `torch.cuda.is_bf16_supported()` returns True and bf16 is 3.6× slower than fp16.** It counts emulation. Ask with `including_emulation=False`.

**A flat constant train/val gap is not overfitting.** Overfitting is a gap that keeps growing while validation loss rises. The pretraining run never overfit; the first real overfitting appeared in the fine-tunes, once the model was on its fourth pass over a few million tokens.
