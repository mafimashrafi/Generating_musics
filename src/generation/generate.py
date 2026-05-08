import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import pretty_midi

from src.models.Transformer_ae import (
    MusicTransformer,
    BOS_TOKEN,
    EOS_TOKEN,
    SEP_TOKEN,
    PAD_TOKEN,
    VOCAB_SIZE,
)

CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "checkpoints"
    / "transformer"
    / "model_final.pt"
)

OUTPUT_DIR = Path("src/generation/generated_midis")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MusicTransformer(
    vocab_size=VOCAB_SIZE,
    d_model=128,
    nhead=4,
    num_layers=3,
    dim_ff=512,
    max_len=2048,
    dropout=0.1,
).to(DEVICE)

state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

model.load_state_dict(state_dict)

model.eval()

print("Model loaded successfully.")

def tokens_to_midi(tokens, out_path):

    midi = pretty_midi.PrettyMIDI()

    instrument = pretty_midi.Instrument(program=0)

    current_time = 0.0
    duration = 0.5

    for token in tokens:

        token = int(token)

        if token in [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN]:
            continue

        if 0 <= token <= 127:

            note = pretty_midi.Note(
                velocity=100,
                pitch=token,
                start=current_time,
                end=current_time + duration
            )

            instrument.notes.append(note)

            current_time += duration

    midi.instruments.append(instrument)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    midi.write(str(out_path))
    
def generate_music(index):

    prompt = torch.tensor([[BOS_TOKEN]], device=DEVICE)

    generated = model.generate(
        prompt=prompt,
        max_new=512,
        temperature=1.0,
        top_k=50,
    )

    tokens = generated[0].cpu().numpy()

    output_file = OUTPUT_DIR / f"generated_{index}.mid"

    tokens_to_midi(tokens, output_file)

    print(f"Saved: {output_file}")


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(5):
        generate_music(i + 1)


if __name__ == "__main__":
    main()