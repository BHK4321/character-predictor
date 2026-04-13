import torch
from torch.utils.data import Dataset
import random
import string

class ProcessedDataset(Dataset):
    def __init__(self, words, max_word_length=45, reveal_ratio=0.5):
        self.words = [word.lower() for word in words if len(word) <= max_word_length]
        self.max_length = max_word_length
        self.reveal_ratio = reveal_ratio
        self.char_to_idx = {char: i+1 for i, char in enumerate(string.ascii_lowercase)}
        self.char_to_idx['_'] = 0  # blank
        self.char_to_idx['PAD'] = 27

    def __len__(self): return len(self.words) * 80

    def __getitem__(self, idx):
        word = self.words[idx % len(self.words)]
        reveal_count = int(len(word) * self.reveal_ratio)
        revealed = random.sample(range(len(word)), reveal_count) if reveal_count > 0 else []

        word_state = [0] * self.max_length
        for pos in revealed: word_state[pos] = self.char_to_idx[word[pos]]

        target_pos, target_chars, position_context, vowels = [], [], [0]*self.max_length, set('aeiou')
        for i in range(len(word)):
            if i not in revealed:
                ctx = 0
                if i > 0 and word_state[i-1] != 0: ctx += 1
                if i < len(word)-1 and word_state[i+1] != 0: ctx += 2
                if ctx:
                    target_pos.append(i)
                    target_chars.append(self.char_to_idx[word[i]])
                    position_context[i] = ctx

        count_blanks = word_state[:len(word)].count(0)
        blank_vowel_next = [0]*self.max_length
        for i in range(len(word)):
            if word_state[i] == 0:
                l = word[i-1] if i > 0 else 'x'
                r = word[i+1] if i < len(word)-1 else 'x'
                if l in vowels or r in vowels:
                    blank_vowel_next[i] = 1

        max_targets = 10
        while len(target_pos) < max_targets:
            target_pos.append(-1)
            target_chars.append(0)

        return {
            'word_state': torch.tensor(word_state, dtype=torch.long),
            'position_context': torch.tensor(position_context, dtype=torch.long),
            'target_positions': torch.tensor(target_pos[:max_targets], dtype=torch.long),
            'target_chars': torch.tensor(target_chars[:max_targets], dtype=torch.long),
            'word_length': torch.tensor(len(word), dtype=torch.long),
            'blank_count': torch.tensor(count_blanks, dtype=torch.long),
            'next_to_vowel': torch.tensor(blank_vowel_next, dtype=torch.float)
        }
