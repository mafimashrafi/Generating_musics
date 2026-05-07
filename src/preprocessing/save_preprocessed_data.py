import pretty_midi
import numpy as np
import os
import pickle
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing.midi_preprocessor import MIDIPreprocessor

class DataSaver:
    def __init__(self, output_dir='data/preprocessed'):
        self.output_dir = Path(output_dir)
        self.setup_directories()
    
    def setup_directories(self):
        (self.output_dir / 'piano_rolls').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'sequences').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'tokens').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'metadata').mkdir(parents=True, exist_ok=True)
        print(f"✓ Output directories created at: {self.output_dir}")
    
    def piano_roll_to_tokens(self, piano_roll):
        tokens = []
        for t in range(piano_roll.shape[0]):
            active_pitches = np.where(piano_roll[t, :] > 0)[0]
            tokens.append(active_pitches.tolist())
        return tokens
    
    def save_piano_roll(self, piano_roll, filename, split='train'):
        path = self.output_dir / 'piano_rolls' / split / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, piano_roll)
        return str(path)
    
    def save_sequence(self, segments, filename, split='train'):
        path = self.output_dir / 'sequences' / split / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(segments, f)
        return str(path)
    
    def save_tokens(self, tokens, filename, split='train'):
        path = self.output_dir / 'tokens' / split / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(tokens, f)
        return str(path)
    
    def process_and_save_file(self, file_path, preprocessor, split='train'):
        filename = Path(file_path).stem

        midi = preprocessor.load_midi(file_path)
        if midi is None:
            return None

        piano_roll = preprocessor.to_piano_roll(midi)
        normalized_pr = preprocessor.normalize_timing(piano_roll)

        segments = preprocessor.segment_sequences(normalized_pr)
        
        tokens = self.piano_roll_to_tokens(normalized_pr)

        pr_path = self.save_piano_roll(piano_roll, f"{filename}_full.npy", split)
        seq_path = self.save_sequence(segments, f"{filename}_segments.pkl", split)
        tok_path = self.save_tokens(tokens, f"{filename}_tokens.json", split)
        
        metadata = {
            'filename': filename,
            'original_file': file_path,
            'split': split,
            'piano_roll_path': pr_path,
            'piano_roll_shape': piano_roll.shape,
            'sequence_path': seq_path,
            'num_segments': len(segments),
            'segment_shape': segments.shape,
            'tokens_path': tok_path,
            'num_timesteps': len(tokens),
            'n_pitches': preprocessor.n_pitches,
            'fs': preprocessor.fs,
            'window_size': preprocessor.window_size
        }
        
        return metadata
    
    def process_and_save_folder(self, folder_path, split='train', max_files=None):
        preprocessor = MIDIPreprocessor(fs=16, window_size=64, pitch_range=(21, 109))
        
        midi_files = sorted(Path(folder_path).glob("*.midi")) + sorted(Path(folder_path).glob("*.mid"))
        if max_files:
            midi_files = midi_files[:max_files]
        
        print(f"\n{'='*70}")
        print(f"Processing {len(midi_files)} {split} files")
        print(f"{'='*70}")
        
        all_metadata = []
        failed_files = []
        
        for idx, file_path in enumerate(midi_files):
            try:
                metadata = self.process_and_save_file(str(file_path), preprocessor, split)
                if metadata:
                    all_metadata.append(metadata)
                    print(f"[{idx+1:4d}/{len(midi_files)}] ✓ {Path(file_path).name:50s} | "
                        f"Segments: {metadata['num_segments']:4d} | "
                        f"Shape: {metadata['segment_shape']}")
                else:
                    failed_files.append(str(file_path))
                    print(f"[{idx+1:4d}/{len(midi_files)}] ✗ Failed: {Path(file_path).name}")
            
            except Exception as e:
                failed_files.append(str(file_path))
                print(f"[{idx+1:4d}/{len(midi_files)}] ✗ Error: {Path(file_path).name} - {str(e)[:50]}")
        
        metadata_path = self.output_dir / 'metadata' / f"{split}_metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        print(f"\n{'-'*70}")
        print(f"✓ Processed: {len(all_metadata)} files")
        print(f"✗ Failed:    {len(failed_files)} files")
        print(f"Metadata saved to: {metadata_path}")
        print(f"{'-'*70}\n")
        
        return all_metadata, failed_files
    
    def generate_summary(self):
        summary = {
            'splits': {},
            'total_files': 0,
            'total_segments': 0
        }
        
        for split in ['train', 'test', 'validation']:
            metadata_path = self.output_dir / 'metadata' / f"{split}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata_list = json.load(f)
                
                total_segments = sum(m['num_segments'] for m in metadata_list)
                summary['splits'][split] = {
                    'num_files': len(metadata_list),
                    'total_segments': total_segments
                }
                summary['total_files'] += len(metadata_list)
                summary['total_segments'] += total_segments
        
        return summary

def main():
    saver = DataSaver(output_dir='data/preprocessed_output')

    splits = {
        'train': 'data/processed/train',
        'test': 'data/processed/test',
        'validation': 'data/processed/validation'
    }
    
    all_results = {}
    
    for split, folder_path in splits.items():
        if Path(folder_path).exists():
            metadata, failed = saver.process_and_save_folder(
                folder_path, 
                split=split, 
                max_files=None  # Set to a number to limit files during testing
            )
            all_results[split] = {
                'processed': len(metadata),
                'failed': len(failed)
            }
        else:
            print(f"⚠ Folder not found: {folder_path}")

    print("\n" + "="*70)
    print("PREPROCESSING SUMMARY")
    print("="*70)
    
    summary = saver.generate_summary()
    
    for split, stats in summary['splits'].items():
        print(f"\n{split.upper()}:")
        print(f"  Files:    {stats['num_files']}")
        print(f"  Segments: {stats['total_segments']}")
    
    print(f"\nTOTAL:")
    print(f"  Files:    {summary['total_files']}")
    print(f"  Segments: {summary['total_segments']}")
    print(f"\nOutput directory: {saver.output_dir}")
    print("="*70)
    
    # Save summary
    summary_path = saver.output_dir / 'metadata' / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary saved to: {summary_path}\n")

if __name__ == "__main__":
    main()