import re
from collections import Counter

class Tokenizer:
    def __init__(self):
        self.word_to_index = {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2}
        self.index_to_word = {0: '<PAD>', 1: '<UNK>', 2: '<EOS>'}
        self.word_count = Counter()
        self.total_words = 0
        self.vocab_size = 3
        self.eos_token_id = 2

    def fit(self, text):
        words = self.preprocess(text).split()
        self.word_count.update(words)
        self.total_words = sum(self.word_count.values())
        
        for word, count in self.word_count.most_common():
            if count > 1 and word not in self.word_to_index:
                self.word_to_index[word] = self.vocab_size
                self.index_to_word[self.vocab_size] = word
                self.vocab_size += 1

    def encode(self, text):
        return [self.word_to_index.get(word, self.word_to_index['<UNK>']) for word in self.preprocess(text).split()]

    def decode(self, indices):
        words = [self.index_to_word.get(idx, '') for idx in indices if idx != self.word_to_index['<UNK>']]
        return ' '.join(words).strip()

    def preprocess(self, text):
        # Convert to lowercase
        text = text.lower()
        # Add spaces around punctuation
        text = re.sub(r'([.,!?])', r' \1 ', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
