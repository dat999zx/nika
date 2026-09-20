import re

WORD_PATTERN = r" ?\w+| ?[^\w\s]+|\s+"
EOT = "<|EOT|>"

# split whole text into words
def split_words(text):
    return re.findall(WORD_PATTERN, text)

# count frequency of each pair
# weight: how many times this word appears, so each pair counts that many times
# ans: pass an existing dict to keep adding into it (used to count across many words)
def get_pair_counts(ids, weight=1, ans=None):
    if ans is None:
        ans = {}

    for i in range(1, len(ids)):
        pair = (ids[i - 1], ids[i])
        if not pair in ans:
            ans[pair] = 0
        ans[pair] += weight

    return ans

# merge the pairs from already learnt merges
def merge(ids, pair, new_id):
    ans = []
    
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            ans.append(new_id)
            i += 1
        else:
            ans.append(ids[i])
        i += 1

    return ans

class BPETokenizer:
    def __init__(self):
        self.merges = {} # (a, b) -> new_id
        self.vocab = [bytes([i]) for i in range(256)]
    
    # verbose: print progress every 100 glues (useful for long runs)
    def train(self, text, vocab_size, verbose=False):
        # count each unique word once: {" the": 100000, " of": 60000, ...}
        word_counts = {}
        for doc in text.split(EOT):
            for word in split_words(doc):
                word_counts[word] = word_counts.get(word, 0) + 1

        # one (byte ids, count) entry per unique word
        words = [(list(word.encode("utf-8")), count) for word, count in word_counts.items()]
        num_glues = vocab_size - 256 - 1 # each merge add 1 glue, theres available bytes so subtract those and 1 for EOT token at the end

        if verbose:
            print(f"{len(words)} unique words, {num_glues} glues to learn")

        for i in range(num_glues):
            pair_counts = {}
            for ids, count in words:
                get_pair_counts(ids, count, pair_counts)
            if not pair_counts: # every word is already a single token, nothing left to glue
                break
            max_pair = max(pair_counts, key=pair_counts.get)

            new_id = self._add_merge(max_pair)
            words = [(merge(ids, max_pair, new_id), count) for ids, count in words]

            if verbose and (i + 1) % 100 == 0:
                print(f"glue {i + 1}/{num_glues}: {self.vocab[new_id]!r} (seen {pair_counts[max_pair]} times)")

        self._add_eot()

    # shared by train and load: record one glue as a recipe (merges) and a label (vocab)
    def _add_merge(self, pair):
        new_id = len(self.vocab) # next free slot
        self.merges[pair] = new_id
        self.vocab.append(self.vocab[pair[0]] + self.vocab[pair[1]])
        return new_id

    # shared by train and load: EOT always takes the slot after the last glue
    def _add_eot(self):
        self.eot_id = len(self.vocab)
        self.vocab.append(EOT.encode("utf-8"))

    # one merge per line, "a b", in the order they were learnt
    # only merges are saved: vocab and eot_id are rebuilt from them in load
    def save(self, path):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for a, b in self.merges:
                f.write(f"{a} {b}\n")

    @classmethod
    def load(cls, path):
        tok = cls()
        with open(path, encoding="utf-8") as f:
            for line in f:
                a, b = line.split()
                tok._add_merge((int(a), int(b)))
        tok._add_eot()
        return tok

    def encode(self, text):
        ids = []
        
        for i, doc in enumerate(text.split(EOT)):
            if i > 0:
                ids.append(self.eot_id)
            for word in split_words(doc):
                ids.extend(self._encode_word(word))
        return ids

    # replay the learnt glues on one word, earliest glue first
    def _encode_word(self, word):
        ids = list(word.encode("utf-8"))

        while True:
            pair_counts = get_pair_counts(ids)

            picked_pair = None
            new_id = None
            for pair in pair_counts:
                if pair not in self.merges: continue
                if not picked_pair or new_id > self.merges[pair]:
                    picked_pair = pair
                    new_id = self.merges[pair]

            if picked_pair is not None:
                ids = merge(ids, picked_pair, new_id)
            else:
                break
        
        return ids

    def decode(self, ids):
        ans = b""
        for i in ids:
            ans += self.vocab[i]
        return ans.decode("utf-8", errors="replace")

if __name__ == "__main__":
    tok = BPETokenizer()
    text = "the cat sat<|EOT|>the dog ran<|EOT|>the end"
    tok.train(text, 256 + 10)
    ids = tok.encode(text)
    print(ids.count(tok.eot_id))          # expect 2
    assert tok.decode(ids) == text
    assert tok.eot_id == 256 + 10 - 1     # EOT is in the last slot
    assert len(tok.vocab) == 256 + 10     # vocab is exactly vocab_size

    # save -> load must give back an identical tokenizer
    import os, tempfile
    path = os.path.join(tempfile.gettempdir(), "tok_test.txt")
    tok.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.merges == tok.merges
    assert loaded.vocab == tok.vocab
    assert loaded.eot_id == tok.eot_id
    assert loaded.encode(text) == ids
    print("all tests passed")
