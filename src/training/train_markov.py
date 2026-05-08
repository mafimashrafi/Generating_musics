import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import random
from collections import defaultdict, Counter

import pretty_midi

DATASET_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "preprocessed_output"
    / "tokens"
    / "train"
)

OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "generation"
    / "markov_generated_midis"
)

PAD_TOKEN = 128
BOS_TOKEN = 129
EOS_TOKEN = 130
EMPTY_TOKEN = 131
SEP_TOKEN = 132

class MarkovChainMusicModel:

    def __init__(self, order=2):

        self.order = order
        self.chain = defaultdict(Counter)
        self.start_states = []

    def train(self, token_files):

        for file_path in token_files:

            with open(file_path, "r") as f:
                data = json.load(f)

            if isinstance(data, dict):
                tokens = data.get("tokens", [])
            else:
                tokens = data

            if len(tokens) <= self.order:
                continue

            self.start_states.append(
                tuple(
                    tuple(t) if isinstance(t, list) else t
                    for t in tokens[:self.order]
                )
            )

            for i in range(len(tokens) - self.order):

                state = tuple(
                    tuple(t) if isinstance(t, list) else t
                    for t in tokens[i:i+self.order]
                )
                next_token = tokens[i+self.order]

                if isinstance(next_token, list):
                    next_token = tuple(next_token)

                self.chain[state][next_token] += 1

    def next_token(self, state):

        if state not in self.chain:
            return None

        counter = self.chain[state]

        tokens = list(counter.keys())
        weights = list(counter.values())

        return random.choices(
            tokens,
            weights=weights,
            k=1
        )[0]

    def generate(self, length=512):

        state = random.choice(self.start_states)

        generated = list(state)

        for _ in range(length - self.order):

            nxt = self.next_token(state)

            if nxt is None:
                break

            generated.append(nxt)

            state = tuple(
                generated[-self.order:]
            )

        return generated

def tokens_to_midi(tokens, output_path):

    midi = pretty_midi.PrettyMIDI()

    instrument = pretty_midi.Instrument(program=0)

    current_time = 0.0
    duration = 0.5

    for token in tokens:

        if token in [
            PAD_TOKEN,
            BOS_TOKEN,
            EOS_TOKEN,
            EMPTY_TOKEN,
            SEP_TOKEN
        ]:
            continue

        # SINGLE NOTE
        if isinstance(token, int):

            if 0 <= token <= 127:

                note = pretty_midi.Note(
                    velocity=100,
                    pitch=token,
                    start=current_time,
                    end=current_time + duration
                )

                instrument.notes.append(note)

        # CHORD / MULTI-NOTE TOKEN
        elif isinstance(token, tuple):

            for pitch in token:

                if 0 <= pitch <= 127:

                    note = pretty_midi.Note(
                        velocity=100,
                        pitch=int(pitch),
                        start=current_time,
                        end=current_time + duration
                    )

                    instrument.notes.append(note)

        current_time += duration

    midi.instruments.append(instrument)

    midi.write(str(output_path))

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    token_files = sorted(
        DATASET_DIR.glob("*.json")
    )

    print(f"Found {len(token_files)} token files")

    if len(token_files) == 0:
        raise ValueError("No token files found.")

    print("Training Markov Chain...")

    model = MarkovChainMusicModel(order=2)

    model.train(token_files)

    print("Generating MIDI files...")

    for i in range(5):

        generated_tokens = model.generate(
            length=512
        )

        output_file = (
            OUTPUT_DIR
            / f"markov_generated_{i+1}.mid"
        )

        tokens_to_midi(
            generated_tokens,
            output_file
        )

        print(f"Saved: {output_file}")

    print("\nDone.")


if __name__ == "__main__":
    main()