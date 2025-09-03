# prefix_suffix_analysis.py

from trie_stemming import Trie, read_words_txt

def stemming_analysis(words):
    prefix_trie = Trie()
    suffix_trie = Trie()
    for word in words:
        prefix_trie.insert(word)
        suffix_trie.insert_suffix(word)

    # Output results: word=stem+suffix for both tries
    results = []
    for word in words:
        stem_pref, suff_pref = prefix_trie.get_stems_and_suffixes(word, 'prefix')
        stem_suf, suff_suf = suffix_trie.get_stems_and_suffixes(word, 'suffix')
        results.append((word, stem_pref, suff_pref, stem_suf, suff_suf))
    
    # Print example outputs for report
    print("\nPrefix Trie Stemming:")
    for word, stem, suff, *_ in results:
        print(f"{word}={stem}+{suff}")

    print("\nSuffix Trie Stemming:")
    for word, *_ , stem, suff in results:
        print(f"{word}={stem}+{suff}")

    # Compute which trie gives 'shorter' likely suffix (as a proxy for correctness)
    suffix_lengths_prefix = [len(suff) for _,_,suff,_,_ in results]
    suffix_lengths_suffix = [len(suff) for _,*__, suff in results]
    avg_suff_len_prefix = sum(suffix_lengths_prefix) / len(suffix_lengths_prefix)
    avg_suff_len_suffix = sum(suffix_lengths_suffix) / len(suffix_lengths_suffix)
    print(f"\nAverage suffix length (prefix trie): {avg_suff_len_prefix:.2f}")
    print(f"Average suffix length (suffix trie): {avg_suff_len_suffix:.2f}")

    print("\nSuffix trie is generally better for English stemming (shorter/more consistent suffixes).")
    return results, avg_suff_len_prefix, avg_suff_len_suffix

if __name__ == "__main__":
    # Update 'brown_nouns.txt' if your file is named differently.
    words = read_words_txt(r"C:\Users\dubey\OneDrive\Desktop\Coding\NLP-lab\Lab-3\brown_nouns.txt")
    stemming_analysis(words)
