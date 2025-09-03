# frequency_analysis.py

import matplotlib.pyplot as plt

def tokenize_corpus(text_file):
    """Loads and tokenizes a text dataset (simple whitespace + lowercasing)."""
    with open(text_file, encoding='utf-8') as f:
        text = f.read()
    # Basic tokenization: split on whitespace/punctuation
    import re
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

def build_freq_dist(tokens):
    """Manually build a frequency distribution dictionary."""
    freq = {}
    for tok in tokens:
        freq[tok] = freq.get(tok, 0) + 1
    return freq

def plot_top_n(freq, n=100, title='Top Words', stopwords=set()):
    """Plot histogram for top N words, excluding stopwords if given."""
    filtered = {w: c for w, c in freq.items() if w not in stopwords}
    top = sorted(filtered.items(), key=lambda x: -x[1])[:n]
    words, counts = zip(*top)
    plt.figure(figsize=(16,6))
    plt.bar(words, counts)
    plt.xticks(rotation=90, fontsize=8)
    plt.title(title)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def identify_stopwords(freq, threshold=0.005):
    """Find stop words by frequency proportion (heuristic)."""
    total = sum(freq.values())
    # Words appearing extremely frequently are likely stopwords
    return {w for w,c in freq.items() if c/total > threshold}

if __name__ == "__main__":
    # Use your tokenized text dataset here
    tokens = tokenize_corpus(r"NLP-lab\Lab1\tokenized_hindi_small.txt")
    freq = build_freq_dist(tokens)
    plot_top_n(freq, n=100, title="Top 100 Most Frequent Words (Raw)")

    # Identify and remove stopwords
    stopw = identify_stopwords(freq, threshold=0.004)
    print("Identified stop words:", stopw)

    # Plot after stopwords removal, at different thresholds
    for thr in [10, 50, 100]:
        filtered = {w: c for w, c in freq.items() if w not in stopw and c >= thr}
        plot_top_n(filtered, n=100, title=f"Top 100 (freq≥{thr}, stopwords removed)")
