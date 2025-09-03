import matplotlib.pyplot as plt
import random
from token_loader import load_tokens
from models import NGramModel
from smoothing import AddOneSmoothing, AddKSmoothing, TokenTypeSmoothing
from sentence_probability import log_probability
import numpy as np
from tqdm import tqdm

def plot_log_probs(log_probs, label):
    print(f"Plotting histogram for {label} smoothing...")
    plt.hist(log_probs, bins=30, alpha=0.5, label=label)
    plt.xlabel('Log Probability')
    plt.ylabel('Count')
    plt.legend()

def main():
    print("Starting main program...")
    tokens = load_tokens(r"NLP-lab\Lab1\tokenized_hindi_small.txt")

    print("Initializing models...")
    models = {n: NGramModel(n) for n in [1, 2, 3, 4]}
    for model in models.values():
        model.train(tokens)

    print("Creating smoothing instances...")
    smoothings = {
        "add_one": AddOneSmoothing(models[2]),  # Bigram example; extend accordingly
        "add_k": AddKSmoothing(models[2], k=0.5),
        "token_type": TokenTypeSmoothing(models[2])
    }

    print("Loading test sentences...")
    with open(r"C:\Users\dubey\OneDrive\Desktop\Coding\NLP-lab\Lab-4\random_news_sentences.txt", "r", encoding="utf-8") as f:
        sentences = [line.strip().split() for line in f.readlines()[:1000]]

    results = {}
    for name, smoothing in smoothings.items():
        print(f"Computing log probabilities with {name} smoothing...")
        log_probs = []
        for sent in tqdm(sentences):
            lp = log_probability(sent, models[2], smoothing)  # Bigram example
            log_probs.append(lp)
        results[name] = log_probs
        plot_log_probs(log_probs, label=name)

    plt.title('Sentence Log Probabilities with Different Smoothing')
    plt.savefig("sentence_probabilities.png")
    print("Plot saved as sentence_probabilities.png")
    plt.show()

    print("Summary statistics:")
    for name, log_probs in results.items():
        print(f"{name}:")
        print(f"- Mean: {np.mean(log_probs):.4f}")
        print(f"- Std: {np.std(log_probs):.4f}")

if __name__ == "__main__":
    main()
