# Unsupervised Neural Network for Multi-Genre Music Generation

**Course:** CSE425/EEE474 Neural Networks **Dataset:** MAESTRO v3 (Classical Piano MIDI, 962 training files)

---


## Project Overview

This project builds unsupervised generative neural networks that learn musical patterns directly from MIDI data — no genre labels required. The models learn to reconstruct and generate piano roll sequences, producing novel MIDI compositions.

The pipeline covers three tasks:

Task 1 : LSTM Autoencoder: 5 MIDI files
Task 2 :Variational Autoencoder (VAE) 8 MIDI files
Task 3 : Transformer (decoder-only) Hard 10 MIDI files

> **Bonus:** Markov Chain baseline model — 5 MIDI files (used for evaluation comparison)

## Repository Structure

    Generating_musics/
    │
    ├── README.md                        current
    ├── requirements.txt                 All Python dependencies
    ├── config.py                        Centralized path and hyperparameter config
    │
    ├── data/
    │   └── processed/
    │       ├── train/                   962 training MIDI files (MAESTRO split)
    │       ├── test/                    177 test MIDI files
    │       └── validation/              137 validation MIDI files
    │
    ├── src/
    │   ├── preprocessing/
    │   │   └── midi_preprocessor.py    Converts MIDI → piano roll segments
    │   │
    │   ├── models/
    │   │   ├── LSTM_encoder.py         Task 1: LSTM Autoencoder architecture
    │   │   ├── vae.py                  Task 2: Variational Autoencoder architecture
    │   │   ├── Transformer_ae.py       Task 3: Transformer decoder architecture
    │   │   ├── dataset.py              Token dataset + collate utilities
    │   │   └── markov_chain.py         Bonus: Markov Chain model
    │   │
    │   ├── training/
    │   │   ├── train_ae.py             Task 1: Training script for LSTM Autoencoder
    │   │   ├── train_vae.py            Task 2: Training script for VAE
    │   │   ├── train_transformer.py    Task 3: Training script for Transformer
    │   │   └── train_markov.py         Bonus: Training + generation script for Markov Chain
    │   │
    │   ├── generation/
    │   │   ├── generated_midis/        Task 3: Transformer-generated MIDI files
    │   │   └── markov_generated_midis/ Bonus: Markov-generated MIDI files
    │   │
    │   └── evaluation/
    │       ├── transformer_eval.py     Task 3: Evaluation script
    │       ├── markov_eval.py          Bonus: Evaluation script
    │       ├── evaluation_results.json Task 3: Transformer evaluation output
    │       └── markov_evaluation_results.json  Bonus: Markov evaluation output
    │
    └── outputs/
        ├── plots/
        │   ├── loss_curve_task1.png    Task 1 reconstruction loss curve (10 epochs)
        │   ├── loss_curve_task2.png    Task 2 VAE loss curve (20 epochs)
        │   └── loss_curve_task3.png    Task 3 perplexity curve
        │
        └── checkpoints/
            └── transformer/
                └── model_final.pt      Final saved Transformer weights
        │
        └── generated_midis/
            ├── task1/                  5 MIDI samples from LSTM Autoencoder
            ├── task2/                  8 MIDI samples from VAE
            └── task3/                  10 MIDI samples from Transformer

---


## File Descriptions

### `src/preprocessing/midi_preprocessor.py`

Converts raw `.midi` files into fixed-length piano roll segments for model training.

- Loads each MIDI file using `pretty_midi`
- Extracts piano roll at `fs=16` (16 time steps per second)
- Clips to the 88-key piano range (MIDI pitches 21–108)
- Binarizes: all non-zero values become 1
- Segments into non-overlapping windows of shape `[64, 88]`
- Filters near-silent windows (fewer than 2% active cells discarded)
- Produces: `143,037` train / `17,948` test / `17,455` validation segments


**Output shape:** `(N, 64, 88)` — N segments, 64 time steps, 88 piano keys

---


### `src/models/LSTM_encoder.py` *(Task 1 — built by Mashrafi)*

Bidirectional LSTM Autoencoder for music reconstruction.

- **Encoder:** 2-layer bidirectional LSTM → latent vector `z` of dimension 64
- **Decoder:** LSTM with teacher forcing → reconstructed piano roll
- **Input/Output shape:** `[batch, 64, 88]`

---


### `src/models/vae.py` *(Task 2- Subrata Baishnab)*

Variational Autoencoder extending Task 1 to support probabilistic generation.

- **Encoder:** 2-layer bidirectional LSTM → `μ` (mean) and `log σ²` (log variance)
- **Reparameterization:** `z = μ + σ ⊙ ε` where `ε ~ N(0, I)`
- **Decoder:** Autoregressive LSTM — generates each time step from the previous output
- **Loss:** Focal Loss (reconstruction) + β·KL divergence
- **Key design decisions:**
  * Focal Loss with `pos_weight=20` — fixes the 97–98% piano roll sparsity problem where plain BCE causes the model to predict silence everywhere
  * `logvar` clamped to `[-4, 4]` — prevents numerical instability
  * KL annealing: β=0 for first 5 epochs, then increases to 0.5 — prevents posterior collapse
  * No sigmoid at decoder output during training — raw logits passed to Focal Loss for numerical stability

---


### `src/models/Transformer_ae.py` *(Task 3 — built by Mashrafi)*

Decoder-only Transformer (GPT-style) for autoregressive token-based music generation.

- **Vocabulary:** 133 tokens — MIDI pitches 0–127 plus special tokens:
  * `PAD_TOKEN = 128`, `BOS_TOKEN = 129`, `EOS_TOKEN = 130`, `EMPTY_TOKEN = 131`, `SEP_TOKEN = 132`
- **Architecture:**
  * Token embedding (`nn.Embedding`, `vocab_size × d_model`)
  * Sinusoidal `PositionalEncoding` with dropout, supporting sequences up to `max_len=4096`
  * Stacked `TransformerEncoderLayer` blocks with Pre-LayerNorm (`norm_first=True`) and GELU activation
  * Causal (upper-triangular) attention mask enforces left-to-right generation
  * Linear output head projecting to `vocab_size` logits
- **Generation (`model.generate`):**
  * Accepts a prompt tensor, samples autoregressively up to `max_new` tokens
  * Temperature scaling + top-k filtering before softmax sampling
  * Stops early on `EOS_TOKEN`
- **Weight init:** Xavier uniform across all 2D parameter tensors
- **Default hyperparameters:** `d_model=256`, `nhead=8`, `num_layers=6`, `dim_ff=1024`, `dropout=0.1`

---


### `src/models/dataset.py`

Token dataset and utilities shared by the Transformer training pipeline.

- **`pianoroll_to_tokens`**: Converts a list of timestep pitch lists to a flat integer token sequence with `BOS`/`EOS` framing. Chords are sorted and separated by `SEP_TOKEN`; silent timesteps emit `EMPTY_TOKEN`.
- **`tokens_to_pianoroll`**: Inverse — reconstructs the list-of-lists piano roll from a flat token sequence.
- **`MidiTokenDataset`**: PyTorch `Dataset` that
  * Loads tokenized JSON files (each a `List[List[int]]` piano roll)
  * Slides a window of `seq_len` tokens with step `stride` across each piece
  * Supports pitch-shift augmentation (±6 semitones) at training time
  * Returns `(src, tgt, pad_mask)` tuples ready for cross-entropy training
- **`collate_fn`**: Stacks variable-length batches into padded tensors.

---


### `src/models/markov_chain.py` *(Bonus baseline)*

n-gram Markov Chain model for symbolic music generation.

- **`order`**: The chain order (default 2) — how many previous tokens determine the next
- **`train(token_files)`**: Reads tokenized JSON files and builds a `state → Counter(next_token)` transition table
- **`generate(length)`**: Samples a random start state from the training corpus and walks the chain, picking weighted-random next tokens until `length` tokens are produced or no transition exists
- **`save` / `load`**: JSON serialization of the chain, order, and start states
- Token format is compatible with `dataset.py` (integer lists or dicts with a `"tokens"` key)

---


### `src/training/train_ae.py` *(Task 1, Subrata Baishnab)*

Training script for the LSTM Autoencoder.

- Loads preprocessed numpy arrays from disk
- Trains for 10 epochs with batch size 128 on GPU
- Loss: `BCELoss` (binary cross entropy)
- Optimizer: Adam, lr=1e-3
- Saves checkpoint after every epoch to `/checkpoints/`
- Saves final loss curve to `outputs/plots/loss_curve_task1.png`


**Training results:**

    Epoch  1 — Train: 0.1816  Test: 0.1622
    Epoch  5 — Train: 0.0831  Test: 0.0579
    Epoch 10 — Train: 0.0612  Test: 0.0416

---


### `src/training/train_vae.py` *(Task 2, Subrata Baishnab)*

Training script for the Variational Autoencoder.

- Loads the same preprocessed numpy arrays as Task 1
- Trains for 20 epochs with batch size 128 on GPU
- Loss: Focal Loss (γ=2.0, pos_weight=20) + β·KL divergence
- KL annealing schedule: β=0 for epochs 1–5, then β increases to 0.5
- Optimizer: Adam, lr=1e-3
- Gradient clipping: max norm 1.0
- Saves checkpoint after every epoch
- Best checkpoint: epoch 6 (lowest reconstruction loss with healthy KL)
- Saves final loss curve to `outputs/plots/loss_curve_task2.png`


**Training results (reconstruction loss):**

    Epoch  1 — Train: 0.0280  Test: 0.0008  KL: 1.66  Beta: 0.00
    Epoch  5 — Train: 0.0002  Test: 0.0000  KL: 1.53  Beta: 0.00
    Epoch  6 — Train: 0.0002  Test: 0.0000  KL: 1.52  Beta: 0.00  ← best checkpoint

---


### `src/training/train_transformer.py` *(Task 3)*

Training script for the Transformer decoder.

- Discovers tokenized JSON files under `data/preprocessed_output/tokens/train` and `validation`
- Builds `MidiTokenDataset` windows with configurable `seq_len` and `stride`
- Supports `torch.compile` (PyTorch ≥ 2.0) for free throughput on compatible hardware
- Mixed-precision training via `torch.cuda.amp.GradScaler` on CUDA
- Cosine LR schedule with linear warmup (`--warmup 200` steps)
- Gradient clipping (`--clip_grad 1.0`), AdamW optimizer (`weight_decay=1e-2`)
- Saves epoch checkpoints and a final `model_final.pt` + `history.json`

**Default fast-training hyperparameters:**

| Argument | Default | Notes |
|---|---|---|
| `--d_model` | 128 | Embedding dimension |
| `--nhead` | 4 | Attention heads |
| `--num_layers` | 3 | Transformer layers |
| `--dim_ff` | 512 | Feed-forward width |
| `--epochs` | 5 | Training epochs |
| `--batch_size` | 64 | Sequences per batch |
| `--seq_len` | 256 | Tokens per window |
| `--stride` | 512 | Window stride (larger = fewer windows) |
| `--lr` | 5e-4 | Peak learning rate |

---


### `src/training/train_markov.py` *(Bonus baseline)*

Combined training + MIDI generation script for the Markov Chain model.

- Reads all tokenized JSON files from `data/preprocessed_output/tokens/train`
- Trains a 2nd-order Markov Chain (handles both plain integer lists and chord tuples)
- Generates 5 MIDI files of 512 tokens each, saved to `src/generation/markov_generated_midis/`
- MIDI conversion: each non-special integer token → a 0.5-second note at velocity 100; chord tuples play all pitches simultaneously

---


### `src/evaluation/transformer_eval.py` and `markov_eval.py`

Evaluation scripts for the Transformer (Task 3) and Markov Chain (Bonus) models. Both share the same three metrics:

| Metric | Description | Ideal |
|---|---|---|
| **Pitch Histogram Similarity** | L1 distance between normalized 12-bin chroma distributions of generated vs. reference | Lower → more similar pitch content |
| **Rhythm Diversity Score** | Ratio of unique note durations to total notes | Higher → more rhythmic variety |
| **Repetition Ratio** | Fraction of 4-note pitch patterns that appear more than once | Lower → less repetitive |

Generated files are paired with reference MIDI files from `data/processed/test/` (cycling if needed). Results are saved as JSON.

---


### `outputs/plots/`

Loss curve plots saved as high-resolution PNG files.

- `loss_curve_task1.png` — Train vs test BCE loss over 10 epochs for LSTM Autoencoder
- `loss_curve_task2.png` — Train vs test Focal Loss over 20 epochs for VAE
- `loss_curve_task3.png` — Perplexity curve for Transformer

---


### `outputs/generated_midis/`

generated samples drive link: <https://drive.google.com/drive/folders/13ariRTp1VWUE4A23zM-JK-zAvSwgUMwE?usp=drive_link> All generated MIDI samples. Each file can be played with MuseScore, VLC, or any online MIDI player (e.g. [onlinesequencer.net](https://onlinesequencer.net)).

- `task1/sample_1.mid` to `sample_5.mid` — Generated by sampling random latent vectors `z ~ N(0,I)` and decoding through the LSTM Autoencoder
- `task2/vae_sample_1.mid` to `vae_sample_8.mid` — Generated by sampling `z ~ N(0,I)` and decoding autoregressively through the VAE (epoch 6 checkpoint)
- `task3/` — 10 long-sequence compositions generated autoregressively by the Transformer
- `markov_generated_midis/markov_generated_1.mid` to `markov_generated_5.mid` — Generated by the Markov Chain baseline

---


## How to Run

### 1. Install dependencies

    pip install -r requirements.txt

### 2. Preprocess data

    from src.preprocessing.midi_preprocessor import MIDIPreprocessor

    processor  = MIDIPreprocessor(fs=16, window_size=64)
    train_data = processor.process_folder("data/processed/train/")
    test_data  = processor.process_folder("data/processed/test/")

### 3. Train Task 1 (LSTM Autoencoder)

    python src/training/train_ae.py

### 4. Train Task 2 (VAE)

    python src/training/train_vae.py

### 5. Train Task 3 (Transformer)

    python src/training/train_transformer.py

### 6. Train & Generate — Markov Chain baseline

    python src/training/train_markov.py

### 7. Evaluate

    # Transformer
    python src/evaluation/transformer_eval.py

    # Markov Chain
    python src/evaluation/markov_eval.py

---


## Model Comparison

| Model | Final Train Loss | Final Test Loss | Notes |
| --- | --- | --- | --- |
| Task 1 — LSTM Autoencoder | 0.0612 | 0.0416 | BCE Loss |
| Task 2 — VAE | 0.0002 | 0.0000 | Focal Loss, epoch 6 checkpoint |
| Task 3 — Transformer | — | Perplexity reported | Cross-entropy |
| Bonus — Markov Chain | — | — | Evaluated via pitch/rhythm/repetition metrics only |

### Evaluation Results

**Transformer (Task 3) — `evaluation_results.json`:**

| File | Pitch Histogram Similarity ↓ | Rhythm Diversity ↑ | Repetition Ratio ↓ |
|---|---|---|---|
| generated_1.mid | 1.2565 | 0.0024 | 0.9289 |
| generated_2.mid | 0.8686 | 0.0048 | 0.3720 |
| generated_3.mid | 1.5263 | 0.0024 | 0.9236 |
| generated_4.mid | 1.3367 | 0.0025 | 0.9003 |
| generated_5.mid | 0.5532 | 0.0027 | 0.6765 |
| **Average** | **1.1083** | **0.0029** | **0.7603** |

**Markov Chain (Bonus) — `markov_evaluation_results.json`:**

| File | Pitch Histogram Similarity ↓ | Rhythm Diversity ↑ | Repetition Ratio ↓ |
|---|---|---|---|
| markov_generated_1.mid | 0.9798 | 0.0006 | 0.8418 |
| markov_generated_2.mid | 0.9387 | 0.0004 | 0.8999 |
| markov_generated_3.mid | 0.8414 | 0.0004 | 0.8993 |
| markov_generated_4.mid | 0.4689 | 0.0005 | 0.8432 |
| markov_generated_5.mid | 0.8135 | 0.0004 | 0.9208 |
| **Average** | **0.8085** | **0.0005** | **0.8810** |

**Key takeaways:**
- The Markov Chain achieves better average pitch histogram similarity (0.81 vs 1.11), matching reference pitch distributions more closely.
- The Transformer shows slightly more rhythmic variety but both models produce very low rhythm diversity scores, reflecting the fixed-duration tokenization used at generation time.
- The Transformer produces less repetitive output on average (0.76 repetition ratio vs 0.88 for Markov), though results vary significantly per file.

---


## Key Implementation Notes

**Why Focal Loss instead of BCE for Task 2?** Piano roll matrices are ~97–98% zeros (silence). Under plain BCE, the model achieves low loss by predicting near-zero everywhere — it never learns to generate notes. Focal Loss with `pos_weight=20` forces the model to treat active note cells as 20× more important than silence, breaking this equilibrium.

**Why KL annealing?** Without annealing, the KL term is active from epoch 1 and penalizes any deviation from the prior before the encoder has learned a useful representation. This causes posterior collapse — the encoder outputs `μ≈0, σ≈1` for all inputs and `z` carries no musical information. Setting β=0 initially lets reconstruction stabilize first.

**Why autoregressive decoding for generation?** At generation time (no input sequence available), the decoder generates step by step — each output is fed as the next input. This is the correct VAE generation procedure and ensures `z` is actually used to condition the output.

**Why Pre-LayerNorm (`norm_first=True`) for the Transformer?** Pre-LN (applying LayerNorm before the attention/FFN sublayers rather than after) improves gradient flow during early training and reduces sensitivity to learning rate, making the model more stable without learning rate warm-up tuning.

**Why a Markov Chain baseline?** The Markov Chain requires no GPU, no gradient computation, and trains in seconds. It serves as a simple sanity-check baseline — if a neural model cannot outperform n-gram statistics on pitch or repetition metrics, the neural model likely hasn't converged or is underfitting.

---


Team

Subrata — Preprocessing pipeline, Task 1 training, Task 2 (VAE) design and training
Mashrafi — Task 1 model architecture (LSTM_encoder.py), Task 3 Transformer, Markov Chain baseline, evaluation scripts

## Requirements

    torch>=2.0
    pretty_midi
    numpy
    matplotlib
    miditok

See `requirements.txt` for full version-pinned list.
