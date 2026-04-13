import torch
import torch.nn as nn
import string
from model import MaskedCharacterPredictor
from utilities import build_lengthwise_frequencies, get_dictionary_filtered_multipliers, get_best_first_guess, clean_state_dict


class Solver:
    def __init__(self, model_path):
        self.model = MaskedCharacterPredictor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dictionary = open("words_250000_train.txt").read().splitlines()
        self.char_to_idx = {c: i+1 for i, c in enumerate(string.ascii_lowercase)}
        self.char_to_idx['_'] = 0
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items() if v != 0}
        self.length_freq = build_lengthwise_frequencies(self.dictionary)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
        # Dictionary-based RL multipliers
        self.dict_multipliers = {letter: 1.0 for letter in string.ascii_lowercase}
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        cleaned_checkpoint0 = clean_state_dict(checkpoint)
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(cleaned_checkpoint0)
        else:
            self.model.load_state_dict(cleaned_checkpoint0)
        self.model.eval()
    def update_dict_multipliers(self, word_pattern, wrong_letters):
        
        multipliers, matching_count, eliminated_count = get_dictionary_filtered_multipliers(
            word_pattern, wrong_letters, self.dictionary
        )
        self.dict_multipliers = multipliers
        return matching_count, eliminated_count

    def predict_letter(self, word_state, guessed_letters=None):
        if guessed_letters is None:
            guessed_letters = set()
        wrong_letters = {ch for ch in guessed_letters if ch not in word_state and ch.isalpha()}
        if wrong_letters is None:
            wrong_letters = set()
        if ' ' in word_state:
            word_state = word_state.replace(' ', '')
        
        word_pattern = list(word_state)
        matching_count, eliminated_count = self.update_dict_multipliers(word_pattern, wrong_letters)
        
        if word_state.count('_') == len(word_state):
            return get_best_first_guess(len(word_state), guessed_letters, self.length_freq)
        
        max_length = 45
        state_indices = []
        position_context = []
    
        for i, char in enumerate(word_state):
            if char == '_':
                state_indices.append(0)
                ctx = 0
                if i > 0 and word_state[i-1] != '_': ctx += 1
                if i < len(word_state)-1 and word_state[i+1] != '_': ctx += 2
                position_context.append(ctx)
            else:
                state_indices.append(self.char_to_idx.get(char, 27))
                position_context.append(0)
    
        while len(state_indices) < max_length:
            state_indices.append(27)
            position_context.append(0)
    
        word_tensor = torch.tensor([state_indices], dtype=torch.long).to(self.device)
        context_tensor = torch.tensor([position_context], dtype=torch.long).to(self.device)
        length_tensor = torch.tensor([len(word_state)], dtype=torch.long).to(self.device)
        blank_count_tensor = torch.tensor([word_state.count('_')], dtype=torch.long).to(self.device)
    
        blank_vowel_next = [0] * max_length
        for i in range(len(word_state)):
            if word_state[i] == '_':
                l = word_state[i-1] if i > 0 else 'x'
                r = word_state[i+1] if i < len(word_state)-1 else 'x'
                if l in 'aeiou' or r in 'aeiou':
                    blank_vowel_next[i] = 1
        blank_vowel_tensor = torch.tensor([blank_vowel_next], dtype=torch.float).to(self.device)
    
        with torch.no_grad():
            model_out = self.model(
                word_tensor, context_tensor, length_tensor,
                blank_count_tensor, blank_vowel_tensor
            )
        # print(self.dict_multipliers)
        # Apply dictionary-based multipliers
        dict_multipliers_tensor = torch.tensor(
            [self.dict_multipliers[chr(ord('a') + i)] for i in range(26)],
            device=self.device
        )
        
        best_predictions = []
        for i in range(len(word_state)):
            if word_state[i] == '_':
                # Apply dictionary multipliers to model predictions
                adjusted_logits = model_out[0, i, :]
                reveal_ratio = sum(1 for c in word_state if c != '_') / len(word_state)
                if reveal_ratio < 0.35:
                    adjusted_logits *= dict_multipliers_tensor
                
                probs = torch.softmax(adjusted_logits, dim=0)
                for j, prob in enumerate(probs):
                    letter = chr(ord('a') + j)
                    if letter not in guessed_letters:
                        best_predictions.append((letter, prob.item(), i))
    
        if best_predictions:
            best_predictions.sort(key=lambda x: x[1], reverse=True)
            return best_predictions[0][0]
    
        # Fallback to frequency-based guess
        available_letters = [c for c in string.ascii_lowercase if c not in guessed_letters]
        if available_letters:
            return available_letters[0]
        return None
