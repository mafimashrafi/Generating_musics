import pretty_midi
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

class MIDIPreprocessor:
    def __init__(self, fs=16, window_size=64, pitch_range=(21, 109)):
        self.fs          = fs
        self.window_size = window_size
        self.pitch_low   = pitch_range[0]
        self.pitch_high  = pitch_range[1]
        self.n_pitches   = pitch_range[1] - pitch_range[0]

    def load_midi(self, file_path):
        try:
            midi = pretty_midi.PrettyMIDI(file_path)
            return midi
        except Exception as e:
            print(f"  Skipping {os.path.basename(file_path)}: {e}")
            return None

    def to_piano_roll(self, midi):
        piano_roll = midi.get_piano_roll(fs=self.fs)
        piano_roll = piano_roll[self.pitch_low:self.pitch_high, :]
        piano_roll = (piano_roll > 0).astype(np.float32)
        return piano_roll

    def normalize_timing(self, piano_roll):
        piano_roll  = piano_roll.T
        total_steps = piano_roll.shape[0]
        remainder   = total_steps % self.window_size
        if remainder != 0:
            piano_roll = piano_roll[:total_steps - remainder, :]
        return piano_roll

    def segment_sequences(self, piano_roll):
        segments = []
        T        = piano_roll.shape[0]
        for start in range(0, T - self.window_size + 1, self.window_size):
            segment = piano_roll[start : start + self.window_size]
            segments.append(segment)
        return np.array(segments)

    def process_file(self, file_path):
        midi = self.load_midi(file_path)
        if midi is None:
            return None
        piano_roll = self.to_piano_roll(midi)
        piano_roll = self.normalize_timing(piano_roll)
        segments   = self.segment_sequences(piano_roll)
        return segments

    def process_folder(self, folder_path, max_files=None):
        all_segments = []
        files        = sorted(Path(folder_path).glob("*.midi"))
        if max_files:
            files = files[:max_files]
        print(f"Processing {len(files)} files from: {folder_path}")
        for i, file_path in enumerate(files):
            segments = self.process_file(str(file_path))
            if segments is not None and len(segments) > 0:
                all_segments.append(segments)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(files)} files done...")
        result = np.concatenate(all_segments, axis=0)
        print(f"  Done! Total segments: {result.shape[0]}, Shape: {result.shape}")
        return result