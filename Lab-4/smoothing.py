print("Loading smoothing classes...")

class AddOneSmoothing:
    def __init__(self, model):
        print("Initializing Add-One smoothing...")
        self.model = model
        self.V = len(model.vocab)

    def prob(self, ngram):
        context = ngram[:-1] if self.model.n > 1 else ()
        return (self.model.ngram_counts[ngram] + 1) / (self.model.context_counts[context] + self.V)

class AddKSmoothing:
    def __init__(self, model, k=0.5):
        print(f"Initializing Add-K smoothing with k={k}...")
        self.model = model
        self.k = k
        self.V = len(model.vocab)

    def prob(self, ngram):
        context = ngram[:-1] if self.model.n > 1 else ()
        return (self.model.ngram_counts[ngram] + self.k) / (self.model.context_counts[context] + self.k * self.V)

class TokenTypeSmoothing:
    def __init__(self, model):
        self.model = model
        self.V = len(model.vocab)

    def prob(self, ngram):
        context = ngram[:-1] if self.model.n > 1 else ()
        # Not a true probability distribution
        return (self.model.ngram_counts[ngram] + self.V) / (self.model.context_counts[context] + self.V)
