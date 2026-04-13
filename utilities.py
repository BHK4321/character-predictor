import string
from collections import Counter, defaultdict
import torch
from tqdm import tqdm
from tqdm.asyncio import tqdm
from model import MaskedCharacterPredictor
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from dataset import ProcessedDataset


def build_lengthwise_frequencies(word_list):
    """Build a map from word length to letter frequency Counter."""
    length_freq = defaultdict(Counter)
    for word in word_list:
        word = word.lower()
        unique_letters = set(word)
        length_freq[len(word)].update(unique_letters)
    return length_freq

def get_best_first_guess(word_length, guessed_letters, length_freq):
    """
    Returns the best frequency-based first guess for a given word length,
    excluding already guessed letters.
    """
    if word_length not in length_freq:
        # fallback to global frequency
        total_counter = Counter()
        for counter in length_freq.values():
            total_counter += counter
        freq = total_counter
    else:
        freq = length_freq[word_length]

    sorted_letters = [letter for letter, _ in freq.most_common() if letter not in guessed_letters]
    return sorted_letters[0] if sorted_letters else None

def word_matches_pattern(word, pattern, wrong_letters):
    """Check if a word matches the current pattern"""
    if len(word) != len(pattern):
        return False
    if pattern.count('_') == len(pattern):
        return True
    # Reject if it contains any wrong letters
    if any(letter in word for letter in wrong_letters):
        return False
    
    # Check position-by-position for pattern match
    for w_char, p_char in zip(word, pattern):
        if p_char != '_' and w_char != p_char:
            return False
        if p_char == '_' and w_char in wrong_letters:
            return False  # Prevent wrong letters in unknown positions
    
    return True


def get_dictionary_filtered_multipliers(word_pattern, wrong_letters, dictionary):
    """
    Filter dictionary based on current pattern and wrong letters,
    then calculate letter frequency penalties based on positional constraints
    around revealed substrings.
    """

    # Find matching words
    matching_words = []
    
    pattern_str = ''.join(word_pattern)
    for word in dictionary:
        if word_matches_pattern(word, word_pattern, wrong_letters):
            matching_words.append(word)
    multipliers = {letter: 0.9 for letter in string.ascii_lowercase}
    for letter in string.ascii_lowercase:
        for word in matching_words:
            for i in range(len(word)):
                if word[i] == letter and word_pattern[i] == '_':
                    multipliers[letter]=1.1
                    break
            if multipliers[letter] == 1.1:
                break
    return multipliers, len(matching_words), len(dictionary) - len(matching_words)

def clean_state_dict(state_dict):
    """Removes 'module.' prefix from multi-GPU trained models if present"""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def train_model1(words, epochs=10, early_stopping_patience=5):
    print("Updated...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MaskedCharacterPredictor()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(opt, patience=2)
    loss_fn = nn.CrossEntropyLoss()

    best = float('inf')
    patience_counter = 0

    for ep in range(epochs):
        # Curriculum: reveal_ratio increases with epoch (starts hard, becomes easier)
        reveal_schedule = [0.93, 0.87, 0.80, 0.80, 0.73, 0.73, 0.67, 0.67, 0.67, 0.60, 0.60, 0.53, 0.53, 0.47, 0.47, 0.40, 0.40, 0.33, 0.33, 0.27]
        reveal_ratio = reveal_schedule[ep] if ep < len(reveal_schedule) else 0.20
        ds = ProcessedDataset(words, reveal_ratio=reveal_ratio)
        train_len = int(0.9 * len(ds))
        tr, val = random_split(ds, [train_len, len(ds)-train_len])
        dl = DataLoader(tr, shuffle=True, pin_memory=True, batch_size=256, num_workers=4)
        vl = DataLoader(val, pin_memory=True, batch_size=256, num_workers=4)

        print(f"\n--- Epoch {ep+1} | Reveal Ratio: {reveal_ratio:.2f} ---", flush=True)
        model.train()
        total_loss = 0
        batch_count = 0

        for i, batch in enumerate(tqdm(dl, desc="Training", ncols=100)):
            opt.zero_grad()
            out = model(batch['word_state'].to(device), batch['position_context'].to(device),
                        batch['word_length'].to(device), batch['blank_count'].to(device),
                        batch['next_to_vowel'].to(device))
            loss, count = 0, 0
            for b in range(out.size(0)):
                target_pos = batch['target_positions'][b].to(device)
                target_char = batch['target_chars'][b].to(device)
                for p, c in zip(target_pos, target_char):
                    if p >= 0 and c > 0:
                        loss += loss_fn(out[b, p], c-1)
                        count += 1
            if count > 0:
                loss = loss / count
                loss.backward()
                opt.step()
                total_loss += loss.item()
                batch_count += 1
            if i % 20 == 0:
                if isinstance(loss, torch.Tensor):
                    print(f"  Batch {i}/{len(dl)} | Loss: {loss.item():.4f}", flush=True)
                else:
                    print(f"  Batch {i}/{len(dl)} | Loss: N/A (no valid targets)", flush=True)

        train_loss = total_loss / batch_count if batch_count > 0 else 0
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in tqdm(vl, desc="Validation", ncols=100):
                out = model(batch['word_state'].to(device), batch['position_context'].to(device),
                            batch['word_length'].to(device), batch['blank_count'].to(device),
                            batch['next_to_vowel'].to(device))
                loss, count = 0, 0
                for b in range(out.size(0)):
                    target_pos = batch['target_positions'][b].to(device)
                    target_char = batch['target_chars'][b].to(device)
                    for p, c in zip(target_pos, target_char):
                        if p >= 0 and c > 0:
                            loss += loss_fn(out[b, p], c-1)
                            count += 1
                if count > 0:
                    val_loss += loss.item() / count
                    val_batches += 1

        val_loss = val_loss / val_batches if val_batches > 0 else 0
        scheduler.step(val_loss)

        if val_loss < best:
            best = val_loss
            patience_counter = 0
            torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), "best_model1.pth")
            print("Model improved and saved.", flush=True)
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}", flush=True)
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {ep+1}", flush=True)
                break

    return model

def simulate(solver1, solver2, word, max_wrong=6, verbose=False):
    """Simulate a game and return results"""
    true_word = word.lower()
    word_state = ['_'] * len(true_word)
    guessed_letters = set()
    wrong_letters = set()
    wrong_count = 0
    
    if verbose:
        print(f"Word: {true_word}")
        print(f"Initial state: {''.join(word_state)}")
    
    while '_' in word_state and wrong_count < max_wrong:
        # Get prediction
        reveal_ratio = len([c for c in word_state if c != '_']) / len(true_word)
        solver = solver1 if reveal_ratio > 0.65 else solver2
        guess = solver.predict_letter(''.join(word_state), guessed_letters)
        
        if guess is None or guess in guessed_letters:
            break
            
        guessed_letters.add(guess)
        
        # Check if guess is correct
        if guess in true_word:
            # Reveal all instances of the letter
            for i, char in enumerate(true_word):
                if char == guess:
                    word_state[i] = char
            if verbose:
                print(f"Correct guess '{guess}': {''.join(word_state)}")
        else:
            wrong_letters.add(guess)
            wrong_count += 1
            if verbose:
                print(f"Wrong guess '{guess}' ({wrong_count}/{max_wrong}): {''.join(word_state)}")
    
    success = '_' not in word_state
    if verbose:
        print(f"Game result: {'WIN' if success else 'LOSE'}")
        print(f"Final state: {''.join(word_state)}")
        print(f"Wrong guesses: {sorted(wrong_letters)}")
    
    return {
        'success': success,
        'word': true_word,
        'guesses': len(guessed_letters),
        'wrong_guesses': wrong_count,
        'final_state': ''.join(word_state)
    }