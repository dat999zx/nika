import re
import os
from datasets import load_dataset
from nika.tokenizer import EOT

DATASET = "benjaminbeilharz/better_daily_dialog"

"""
Dataset({
    features: ['dialog_id', 'utterance', 'turn_type', 'emotion'],
    num_rows: 87170
})

0  {'dialog_id': 0, 'utterance': 'Say , Jim , how about going for a few beers after dinner ? ', 'turn_type': 3, 'emotion': 0}
1  {'dialog_id': 0, 'utterance': ' You know that is tempting but is really not good for our fitness . ', ...}
2  {'dialog_id': 0, 'utterance': ' What do you mean ? It will help us to relax . ', ...}
3  {'dialog_id': 0, 'utterance': " Do you really think so ? I don't . ...", ...}
4  {'dialog_id': 0, 'utterance': " I guess you are right.But what shall we do ? ...", ...}


dialog_id: to know if its same conversation
uterrance: the text
other 2 col ignore

need to clean up dataset first:
- strip the 2 ends extra spaces
- clean up extra spaces between letters: "mean ? It"
- missing space "right.But"
"""

def clean(text):
    """'Say , Jim , ... dinner ? ' -> 'Say, Jim, ... dinner?'

    the dataset uses the curly apostrophe and spaces it out on both sides,
    so "he ' s" needs both halves closed up: "he's"
    """
    text = text.strip()
    text = re.sub(r"\s+([,.!?;:’'])", r"\1", text)   # no space BEFORE punctuation
    text = re.sub(r"([’'])\s+(?=\w)", r"\1", text)   # no space AFTER an apostrophe
    return re.sub(r"\s+", " ", text)

def dialogues(split):
    ds = load_dataset(DATASET, split=split)
    turns, current = [], None

    for row in ds:
        if row["dialog_id"] != current and turns:
            yield turns
            turns = []
        current = row["dialog_id"]
        turns.append(clean(row["utterance"]))

    if turns:
        yield turns

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    seen = set()
    n_written = n_turns = n_dupes = 0

    with open("data/dialog.txt", "w", encoding="utf-8", newline="\n") as f:
        for split in ("train", "validation", "test"):
            for turns in dialogues(split):
                if len(turns) < 2: # skip 1 line conversation
                    continue

                lines = []
                for i, t in enumerate(turns):
                    if i % 2 == 0:
                        lines.append("A: " + t)
                    else:
                        lines.append("B: " + t)
                text = "\n".join(lines)
                
                if text in seen: # check for dupe
                    n_dupes += 1
                    continue
                seen.add(text)

                f.write(text + "\n" + EOT + "\n") # add EOT to each conversation
                n_written += 1
                n_turns += len(turns)

    size_mb = os.path.getsize("data/dialog.txt") / 1e6
    print(f"{n_written} dialogues, {n_turns} turns, {n_dupes} duplicates dropped, {size_mb:.1f} MB")
