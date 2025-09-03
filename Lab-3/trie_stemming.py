# trie_stemming.py

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.words = []  # For suffix trie analysis

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        """Insert word into prefix trie."""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def insert_suffix(self, word):
        """Insert word into suffix trie (words reversed)."""
        node = self.root
        for char in reversed(word):
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.words.append(word)  # Record which words pass through this node
        node.is_end = True

    def get_stems_and_suffixes(self, word, trie_type='prefix'):
        """
        Find the likely stem and suffix by following the path with max branching.
        Returns (stem, suffix) pairs.
        """
        node = self.root
        max_branch_node = self.root
        max_depth = 0
        path = []
        chars = word if trie_type == 'prefix' else reversed(word)
        for i, char in enumerate(chars):
            path.append(char)
            if char in node.children:
                node = node.children[char]
                # Branching point is where largest number of children
                if len(node.children) > len(max_branch_node.children):
                    max_branch_node = node
                    max_depth = i+1
            else:
                break
        # For prefix: stem = up to max_depth, suffix = the rest
        # For suffix: suffix = up to max_depth, stem = the rest (reverse back)
        if trie_type == 'prefix':
            stem = word[:max_depth]
            suffix = word[max_depth:]
        else:
            word_rev = word[::-1]
            suffix = word_rev[:max_depth][::-1]
            stem = word_rev[max_depth:][::-1]
        return stem, suffix

def read_words_txt(txt_file):
    """Read brown_nouns.txt, one word per line."""
    with open(txt_file, encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    return words
