import numpy as np
from tqdm import tqdm

def sentence_probability(sentence_tokens, model, smoothing=None):
    prob = 1.0
    n = model.n
    padded = ['<s>'] * (n - 1) + sentence_tokens + ['</s>']
    for i in range(len(padded) - n + 1):
        ngram = tuple(padded[i:i+n])
        if smoothing:
            p = smoothing.prob(ngram)
        else:
            p = model.prob(ngram)
        prob *= p
    return prob

def log_probability(sentence_tokens, model, smoothing=None):
    logprob = 0.0
    n = model.n
    padded = ['<s>'] * (n - 1) + sentence_tokens + ['</s>']
    for i in range(len(padded) - n + 1):
        ngram = tuple(padded[i:i+n])
        if smoothing:
            p = smoothing.prob(ngram)
        else:
            p = model.prob(ngram)
        logprob += np.log(p) if p > 0 else -np.inf
    return logprob
