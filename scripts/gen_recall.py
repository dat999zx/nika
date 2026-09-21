"""Facts stated in the conversation, then asked about.

This tests the one "thinking" skill a small model does well: look back in the
context and copy the right thing. Mechanistically it is an induction head, which
emerges in transformers with as few as two attention layers.

The entity pools are DISJOINT between train and heldout, so the test asks about
names and cities the model has never seen. Memorising pairs cannot pass that.
"""
import argparse
import os
import random

from nika.tokenizer import EOT

SEPARATOR = "\n" + EOT + "\n"

NAMES = ("Sam Anna Tom Lisa Ben Maria Jack Emma Leo Nina Omar Sara Paul Ruby Max Ella "
         "Nick Zoe Adam Mia Luke Iris Finn Rosa Hugo Lena Ivan Tara Noah Jade").split()
HELD_NAMES = "Oscar Freya Dmitri Yuki Pablo Ingrid Kofi Mei Rafael Astrid".split()

CITIES = ("Oslo Tokyo Paris Cairo Lima Dublin Seoul Madrid Boston Vienna Lagos Prague "
          "Miami Zurich Perth Quito Bergen Kyoto Turin Malmo").split()
HELD_CITIES = "Reykjavik Nairobi Osaka Bogota Helsinki".split()

JOBS = ("teacher nurse driver chef writer farmer doctor plumber dentist pilot "
        "lawyer painter baker tailor guard").split()
FOODS = ("pizza sushi curry pasta soup bread rice noodles salad cheese").split()
PETS = ("dog cat rabbit parrot hamster turtle").split()

# fact type -> (statement, question, answer) templates
FACT_TYPES = [
    ("name", "my name is {v}.", "what is my name?", "Your name is {v}."),
    ("city", "I live in {v}.", "where do I live?", "You live in {v}."),
    ("job", "I work as a {v}.", "what do I do?", "You work as a {v}."),
    ("food", "my favourite food is {v}.", "what is my favourite food?", "Your favourite food is {v}."),
    ("pet", "I have a {v}.", "what pet do I have?", "You have a {v}."),
]

ACKS = ["Nice to meet you.", "That's interesting.", "Good to know.", "I see.",
        "Okay.", "Sounds good.", "Got it."]


def pools(heldout):
    """disjoint entity pools, so the test uses values never seen in training"""
    return {
        "name": HELD_NAMES if heldout else NAMES,
        "city": HELD_CITIES if heldout else CITIES,
        "job": JOBS, "food": FOODS, "pet": PETS,
    }


def conversation(heldout):
    pool = pools(heldout)
    kinds = random.sample(FACT_TYPES, random.randint(2, 4))
    values = {name: random.choice(pool[name]) for name, *_ in kinds}

    turns = []
    for name, statement, _, _ in kinds:
        turns.append("A: " + statement.format(v=values[name]))
        turns.append("B: " + random.choice(ACKS))

    # a distractor, so the answer is not simply "the only name in the context"
    if random.random() < 0.4:
        other, city = random.choice(pool["name"]), random.choice(pool["city"])
        turns.append(f"A: my friend {other} lives in {city}.")
        turns.append("B: " + random.choice(ACKS))

    # ask about 1-2 of the facts, sometimes the earliest one, so the distance varies
    asked = random.sample(kinds, random.randint(1, min(2, len(kinds))))
    for name, _, question, answer in asked:
        turns.append("A: " + question)
        turns.append("B: " + answer.format(v=values[name]))

    return "\n".join(turns)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--n-heldout", type=int, default=500)
    p.add_argument("--out", default="data/recall.txt")
    p.add_argument("--heldout", default="data/recall_heldout.txt")
    args = p.parse_args()

    random.seed(0)
    os.makedirs("data", exist_ok=True)

    for path, n, heldout in ((args.out, args.n, False), (args.heldout, args.n_heldout, True)):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for _ in range(n):
                f.write(conversation(heldout) + SEPARATOR)
        print(f"{n} conversations -> {path} ({os.path.getsize(path) / 1e6:.2f} MB)")

    print(f"train entities: {len(NAMES)} names, {len(CITIES)} cities")
    print(f"held-out entities: {len(HELD_NAMES)} names, {len(HELD_CITIES)} cities (disjoint)")
