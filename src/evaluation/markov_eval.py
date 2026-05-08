import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
from collections import Counter

import numpy as np
import pretty_midi


GENERATED_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "generation"
    / "markov_generated_midis"
)

REFERENCE_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "test"
)

def midi_to_features(midi_path):

    midi = pretty_midi.PrettyMIDI(str(midi_path))

    notes_data = []

    for instrument in midi.instruments:

        if instrument.is_drum:
            continue

        for note in instrument.notes:

            notes_data.append(
                (
                    note.start,
                    note.pitch % 12,
                    round(note.end - note.start, 3)
                )
            )
    notes_data.sort(key=lambda x: x[0])

    pitches = [x[1] for x in notes_data]
    durations = [x[2] for x in notes_data]

    return pitches, durations



def pitch_histogram_similarity(pitches_gen, pitches_ref):

    hist_gen = np.zeros(12)
    hist_ref = np.zeros(12)

    for p in pitches_gen:
        hist_gen[p] += 1

    for p in pitches_ref:
        hist_ref[p] += 1

    hist_gen /= max(hist_gen.sum(), 1)
    hist_ref /= max(hist_ref.sum(), 1)

    return np.sum(np.abs(hist_gen - hist_ref))



def rhythm_diversity_score(durations):

    if len(durations) == 0:
        return 0.0

    return len(set(durations)) / len(durations)



def repetition_ratio(pitches, pattern_size=4):

    if len(pitches) < pattern_size:
        return 0.0

    patterns = []

    for i in range(len(pitches) - pattern_size + 1):
        patterns.append(tuple(pitches[i:i + pattern_size]))

    total_patterns = len(patterns)

    counter = Counter(patterns)

    repeated = sum(v for v in counter.values() if v > 1)

    return repeated / total_patterns



def evaluate_single(generated_midi, reference_midi):

    gen_pitches, gen_durations = midi_to_features(generated_midi)
    ref_pitches, ref_durations = midi_to_features(reference_midi)

    return {
        "generated_file": generated_midi.name,
        "reference_file": reference_midi.name,
        "pitch_histogram_similarity": float(
            pitch_histogram_similarity(gen_pitches, ref_pitches)
        ),
        "rhythm_diversity_score": float(
            rhythm_diversity_score(gen_durations)
        ),
        "repetition_ratio": float(
            repetition_ratio(gen_pitches)
        )
    }



def evaluate_all():

    generated_files = sorted(
        list(GENERATED_DIR.glob("*.mid")) +
        list(GENERATED_DIR.glob("*.midi"))
    )

    reference_files = sorted(
        list(REFERENCE_DIR.glob("*.mid")) +
        list(REFERENCE_DIR.glob("*.midi"))
    )

    if len(reference_files) == 0:
        raise ValueError("No reference MIDI files found.")

    if len(generated_files) == 0:
        raise ValueError("No generated MIDI files found.")

    results = []

    for i, gen_file in enumerate(generated_files):

        ref_file = reference_files[i % len(reference_files)]

        result = evaluate_single(gen_file, ref_file)

        results.append(result)

        print("\n====================================")
        print(f"Generated: {gen_file.name}")
        print(f"Reference: {ref_file.name}")
        print("------------------------------------")
        print(
            f"Pitch Histogram Similarity: "
            f"{result['pitch_histogram_similarity']:.4f}"
        )
        print(
            f"Rhythm Diversity Score: "
            f"{result['rhythm_diversity_score']:.4f}"
        )
        print(
            f"Repetition Ratio: "
            f"{result['repetition_ratio']:.4f}"
        )

    output_file = (
        Path(__file__).resolve().parent
        / "markov_evaluation_results.json"
    )

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved evaluation results to: {output_file}")


if __name__ == "__main__":
    evaluate_all()