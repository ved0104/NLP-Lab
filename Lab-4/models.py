from collections import Counter
from tqdm import tqdm

class NGramModel:
    def __init__(self, n):
        self.n = n
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.vocab = set()

    def train(self, tokens):
        print(f"Training {self.n}-gram model on {len(tokens)} tokens...")
        padded = ['<s>'] * (self.n - 1) + tokens + ['</s>']
        for i in tqdm(range(len(padded) - self.n + 1)):
            ngram = tuple(padded[i:i+self.n])
            context = tuple(padded[i:i+self.n-1]) if self.n > 1 else ()
            self.ngram_counts[ngram] += 1
            self.context_counts[context] += 1
            self.vocab.update(ngram)
        print(f"{self.n}-gram training complete.")
        
    def prob(self, ngram):
        context = ngram[:-1] if self.n > 1 else ()
        ngram_count = self.ngram_counts[ngram]
        context_count = self.context_counts[context] if self.n > 1 else sum(self.ngram_counts.values())
        return ngram_count / context_count if context_count > 0 else 0.0
