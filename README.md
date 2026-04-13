# NLP-Based Masked Character Predictor

Character-level NLP model for masked letter inference, combined with dictionary-aware heuristics and curriculum-based training.

## Architecture Overview

The pipeline predicts missing characters in partially observed words using contextual sequence features and dictionary priors.

### 1. Dataset (`ProcessedDataset` in `dataset.py`)

- Generates synthetic partially revealed word states from a list of words.
- Encodes letters as:
   - `0` -> blank (`_`)
   - `1..26` -> `a..z`
   - `27` -> `PAD`
- Produces tensors used in training:
   - `word_state`
   - `position_context`
   - `target_positions`
   - `target_chars`
   - `word_length`
   - `blank_count`
   - `next_to_vowel`

### 2. Model (`MaskedCharacterPredictor` in `model.py`)

- Character embedding + context embedding.
- CNN branch for local pattern extraction.
- BiLSTM encoder for sequence context.
- Position prior MLP for per-position letter prior.
- Three context-specific decoders:
   - left context
   - right context
   - both sides context
- Output shape: `[batch, max_len, 26]` logits over letters `a..z`.

### 3. Solver (`Solver` in `solver.py`)

- Loads `MaskedCharacterPredictor` weights from a `.pth` checkpoint.
- Uses dictionary filtering (`words_250000_train.txt`) to maintain letter multipliers.
- Combines model output with frequency heuristics.
- Handles cold-start prediction via word-length letter frequency statistics.

## Core Utilities (`utilities.py`)

- `train_model1(words, epochs=10, early_stopping_patience=5)`
   - Trains model with reveal-ratio curriculum schedule.
   - Uses validation loss, LR scheduling, and early stopping.
   - Saves best checkpoint to `best_model1.pth`.

- `simulate_hangman_game(solver1, solver2, word, max_wrong=6, verbose=False)`
   - Simulates one masked-character inference episode.
   - Dynamically switches solver: `solver1` when reveal ratio is high, otherwise `solver2`.
   - Returns a result dict with success, guesses, wrong guesses, and final state.

  Note: the function name is legacy; behavior is generic masked-character solving.

- Heuristic helpers:
   - `build_lengthwise_frequencies`
   - `get_best_first_guess`
   - `word_matches_pattern`
   - `get_dictionary_filtered_multipliers`
   - `clean_state_dict`

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Run solver inference

```python
from solver import Solver

solver = Solver("best_model1.pth")
guess = solver.predict_letter("_e__a___")
print("Next guess:", guess)
```

### 2. Simulate a full round

```python
from solver import Solver
from utilities import simulate_hangman_game

solver1 = Solver("best_model1.pth")
solver2 = Solver("best_model2.pth")

result = simulate_hangman_game(solver1, solver2, "pattern", verbose=True)
```

### 3. Typical use case

- Input pattern: `_ e _ _ a _ _`
- Model objective: infer the most probable next character.
- Strategy: combine neural logits with dictionary-constrained multipliers.

## Notes

- `Solver` expects `words_250000_train.txt` to exist in the project root.
- If your checkpoint was saved from DataParallel training, `clean_state_dict` handles prefix cleanup.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
