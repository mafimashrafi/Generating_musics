# Generating_musics
This project is a walkthrough and implementation of different unsupervised technics to generate multi-genre music. 
# Unsupervised Neural Network for Multi-Genre Music Generation

**Course:** CSE425/EEE474 Neural Networks
**Dataset:** MAESTRO v3 (Classical Piano MIDI, 962 training files)

---

## Project Overview

This project builds unsupervised generative neural networks that learn musical patterns directly from MIDI data — no genre labels required. The models learn to reconstruct and generate piano roll sequences, producing novel MIDI compositions.

The pipeline covers three tasks:


 Task 1 : LSTM Autoencoder: 5 MIDI files 
Task 2 :Variational Autoencoder (VAE) 8 MIDI files 
Task 3 : Transformer (decoder-only)  Hard  10 MIDI files



## Repository Structure

```
Generating_musics/
│
├── README.md                        current
├── requirements.txt                 All Python dependencies
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
│   │   └── transformer.py          Task 3: Transformer decoder architecture
│   │
│   └── training/
│       ├── train_ae.py             Task 1: Training script for LSTM Autoencoder
│       ├── train_vae.py            Task 2: Training script for VAE
│       └── train_transformer.py    ask 3: Training script for Transformer
│
└── outputs/
    ├── plots/
    │   ├── loss_curve_task1.png    Task 1 reconstruction loss curve (10 epochs)
    │   ├── loss_curve_task2.png    Task 2 VAE loss curve (20 epochs)
    │   └── loss_curve_task3.png    Task 3 perplexity curve
    │
    └── generated_midis/
        ├── task1/                  5 MIDI samples from LSTM Autoencoder
        ├── task2/                  8 MIDI samples from VAE
        └── task3/                  10 MIDI samples from Transformer
```

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
  - Focal Loss with `pos_weight=20` — fixes the 97–98% piano roll sparsity problem where plain BCE causes the model to predict silence everywhere
  - `logvar` clamped to `[-4, 4]` — prevents numerical instability
  - KL annealing: β=0 for first 5 epochs, then increases to 0.5 — prevents posterior collapse
  - No sigmoid at decoder output during training — raw logits passed to Focal Loss for numerical stability

---

### `src/models/transformer.py` *(Task 3 — built by Mashrafi)*
Decoder-only Transformer (GPT-style) for autoregressive music generation.

- Token embedding + positional embedding
- Stacked Transformer decoder layers with causal (masked) self-attention
- Predicts next token given all previous tokens: `p(xₜ | x<t)`
- Evaluated using perplexity: `exp(1/T · L_TR)`

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
```
Epoch  1 — Train: 0.1816  Test: 0.1622
Epoch  5 — Train: 0.0831  Test: 0.0579
Epoch 10 — Train: 0.0612  Test: 0.0416
```

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
```
Epoch  1 — Train: 0.0280  Test: 0.0008  KL: 1.66  Beta: 0.00
Epoch  5 — Train: 0.0002  Test: 0.0000  KL: 1.53  Beta: 0.00
Epoch  6 — Train: 0.0002  Test: 0.0000  KL: 1.52  Beta: 0.00  ← best checkpoint
```

---

### `src/training/train_transformer.py` *(Task 3)*
Training script for the Transformer decoder.

- Tokenizes MIDI files using `miditok` REMI scheme
- Trains autoregressively on token sequences
- Loss: negative log-likelihood cross entropy
- Reports perplexity on validation set each epoch

---

### `outputs/plots/`
Loss curve plots saved as high-resolution PNG files.

- `loss_curve_task1.png` — Train vs test BCE loss over 10 epochs for LSTM Autoencoder
- `loss_curve_task2.png` — Train vs test Focal Loss over 20 epochs for VAE
- `loss_curve_task3.png` — Perplexity curve for Transformer

---

### `outputs/generated_midis/`
generated samples drive link: https://drive.google.com/drive/folders/13ariRTp1VWUE4A23zM-JK-zAvSwgUMwE?usp=drive_link 
All generated MIDI samples. Each file can be played with MuseScore, VLC, or any online MIDI player (e.g. [onlinesequencer.net](https://onlinesequencer.net)).

- `task1/sample_1.mid` to `sample_5.mid` — Generated by sampling random latent vectors `z ~ N(0,I)` and decoding through the LSTM Autoencoder
- `task2/vae_sample_1.mid` to `vae_sample_8.mid` — Generated by sampling `z ~ N(0,I)` and decoding autoregressively through the VAE (epoch 6 checkpoint)
- `task3/` — 10 long-sequence compositions generated autoregressively by the Transformer

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess data
```python
from src.preprocessing.midi_preprocessor import MIDIPreprocessor

processor  = MIDIPreprocessor(fs=16, window_size=64)
train_data = processor.process_folder("data/processed/train/")
test_data  = processor.process_folder("data/processed/test/")
```

### 3. Train Task 1 (LSTM Autoencoder)
```bash
python src/training/train_ae.py
```

### 4. Train Task 2 (VAE)
```bash
python src/training/train_vae.py
```

### 5. Train Task 3 (Transformer)
```bash
python src/training/train_transformer.py
```

---

## Model Comparison

| Model | Final Train Loss | Final Test Loss | Notes |
|-------|-----------------|-----------------|-------|
| Task 1 — LSTM Autoencoder | 0.0612 | 0.0416 | BCE Loss |
| Task 2 — VAE | 0.0002 | 0.0000 | Focal Loss, epoch 6 checkpoint |
| Task 3 — Transformer | — | Perplexity reported | Cross-entropy |

---

## Key Implementation Notes

**Why Focal Loss instead of BCE for Task 2?**
Piano roll matrices are ~97–98% zeros (silence). Under plain BCE, the model achieves low loss by predicting near-zero everywhere — it never learns to generate notes. Focal Loss with `pos_weight=20` forces the model to treat active note cells as 20× more important than silence, breaking this equilibrium.

**Why KL annealing?**
Without annealing, the KL term is active from epoch 1 and penalizes any deviation from the prior before the encoder has learned a useful representation. This causes posterior collapse — the encoder outputs `μ≈0, σ≈1` for all inputs and `z` carries no musical information. Setting β=0 initially lets reconstruction stabilize first.

**Why autoregressive decoding for generation?**
At generation time (no input sequence available), the decoder generates step by step — each output is fed as the next input. This is the correct VAE generation procedure and ensures `z` is actually used to condition the output.

---
Team

Subrata — Preprocessing pipeline, Task 1 training, Task 2 (VAE) design and training
Mashrafi — Task 1 model architecture (LSTM_encoder.py), Task 3 Transformer
## Requirements

```
torch>=2.0
pretty_midi
numpy
matplotlib
miditok
```

See `requirements.txt` for full version-pinned list.

---
