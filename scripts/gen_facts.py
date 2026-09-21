"""A glossary from Simple English Wikipedia: "what is X?" -> its first sentence.

Simple Wikipedia is written in deliberately plain English and nearly every article
opens with a definition, so the first sentence is usable as an answer with no
rewriting. This is pure memorisation, which is fine: a term has one fixed answer.

Only the longest articles are kept, as a rough proxy for "well known" - a 23M model
cannot memorise 250k definitions, and the long tail would only blur the common ones.
"""
import argparse
import os
import random
import re

from datasets import load_dataset

from nika.tokenizer import EOT

SEPARATOR = "\n" + EOT + "\n"
MAX_ANSWER_WORDS = 22

QUESTIONS = [
    "what is {t}?", "what's {t}?", "tell me about {t}", "what does {t} mean?",
    "do you know what {t} is?", "can you explain {t}?", "explain {t}",
    "I don't know what {t} is",
]
# no "who is {t}?" - it produced "who is 2021?", and there is no cheap way to
# tell a person from a thing


def first_sentence(text):
    text = re.sub(r"\s*\([^)]*\)", "", text)  # drop parentheticals: dates, pronunciations
    text = re.sub(r"\s+", " ", text)          # newlines to spaces FIRST: articles end
                                              # sentences with ".\n", which ". " misses
    return re.split(r"(?<=[.!?]) ", text)[0].strip().rstrip(".")


def usable(title, sentence):
    words = sentence.split()
    return (
        4 <= len(words) <= MAX_ANSWER_WORDS
        and len(title.split()) <= 3                 # short, askable titles
        and title.lower() in sentence.lower()       # a real definition, not a fragment
        and not title.startswith(("List of", "Index of"))
        and not re.fullmatch(r"[\d\W]+", title)     # years and symbols are not glossary terms
        and " ." not in sentence                    # leftover from a stripped parenthetical
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--terms", type=int, default=4000, help="how many definitions to keep")
    p.add_argument("--per-term", type=int, default=3, help="question phrasings per term")
    p.add_argument("--out", default="data/facts.txt")
    p.add_argument("--corpus", default="data/tiny.txt",
                   help="ranks terms by how often they appear here")
    args = p.parse_args()

    random.seed(0)
    os.makedirs("data", exist_ok=True)

    # article length is a poor proxy for "worth knowing" - it surfaces things like
    # "Deaths in 2013". Word frequency in the pretraining corpus is much closer to
    # what someone would actually ask about, and the corpus is already on disk.
    from collections import Counter
    print(f"counting word frequencies in {args.corpus}...", flush=True)
    freq = Counter(re.findall(r"[a-z]+", open(args.corpus, encoding="utf-8").read(30_000_000).lower()))

    # "The 88" would otherwise score as the frequency of "the", which is enormous
    STOPWORDS = {"the", "of", "a", "an", "and", "in", "on", "at", "to", "for", "is"}
    # Simple Wikipedia has articles titled "Not", "You", "May". They are common
    # words, not things anyone asks the meaning of, so drop titles made only of them.
    TOO_COMMON = {w for w, _ in freq.most_common(400)}

    def commonness(title):
        words = [w for w in re.findall(r"[a-z]+", title.lower()) if w not in STOPWORDS]
        if not words or any(len(w) < 3 for w in words):
            return 0
        if all(w in TOO_COMMON for w in words):
            return 0
        return min(freq[w] for w in words) # a phrase is as common as its rarest word

    ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train", streaming=True)
    candidates = []
    for n, row in enumerate(ds, 1):
        sentence = first_sentence(row["text"])
        if usable(row["title"], sentence):
            candidates.append((commonness(row["title"]), row["title"], sentence))
        if n % 50_000 == 0:
            print(f"{n} articles scanned, {len(candidates)} usable", flush=True)

    candidates.sort(reverse=True)
    kept = candidates[:args.terms]

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for _, title, sentence in kept:
            for question in random.sample(QUESTIONS, args.per_term):
                f.write(f"A: {question.format(t=title)}\nB: {sentence}." + SEPARATOR)

    print(f"{len(candidates)} usable of {n} articles; kept {len(kept)} terms "
          f"x {args.per_term} phrasings -> {args.out} "
          f"({os.path.getsize(args.out) / 1e6:.2f} MB)")
    print("examples:", ", ".join(t for _, t, _ in kept[:10]))
