from tqdm import tqdm

def load_tokens(file_path):
    print(f"Loading tokens from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
        tokens = data.strip().split()
    print(f"Loaded {len(tokens)} tokens.")
    return tokens

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python token_loader.py <tokenized_txt_file>")
    else:
        tokens = load_tokens(sys.argv[1])
        print("First 20 tokens preview:", tokens[:20])