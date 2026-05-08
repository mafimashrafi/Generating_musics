from pathlib import Path

ROOT = Path(__file__).parent.resolve()

DATA_ROOT        = ROOT / "data" / "preprocessed_output"
TOKEN_DIR        = DATA_ROOT / "tokens"
TOKEN_TRAIN_DIR  = TOKEN_DIR / "train"
TOKEN_VAL_DIR    = TOKEN_DIR / "validation"
TOKEN_TEST_DIR   = TOKEN_DIR / "test"
SEQ_TRAIN_DIR    = DATA_ROOT / "sequences" / "train"
SEQ_TEST_DIR     = DATA_ROOT / "sequences" / "test"

OUTPUT_ROOT      = ROOT / "outputs"
MIDI_OUT_DIR     = OUTPUT_ROOT / "generated_midis"
PLOT_DIR         = OUTPUT_ROOT / "plots"
MODEL_CKPT_DIR   = OUTPUT_ROOT / "checkpoints" / "transformer"
LSTM_CKPT_DIR    = OUTPUT_ROOT / "checkpoints" / "lstm"

TRANSFORMER = dict(
    vocab_size  = 133,   # 0-127 pitches + PAD/BOS/EOS/EMPTY/SEP
    d_model     = 256,
    nhead       = 8,
    num_layers  = 6,
    dim_ff      = 1024,
    max_len     = 4096,
    dropout     = 0.1,
)

TRANSFORMER_TRAIN = dict(
    epochs      = 10,
    batch_size  = 16,
    seq_len     = 512,
    stride      = 256,
    lr          = 3e-4,
    warmup      = 500,
    clip_grad   = 1.0,
    augment     = True,
    save_every  = 5,
)