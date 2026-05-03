import pandas as pd
import pretty_midi
import os
import shutil
from pathlib import Path

CSV_FILE = "maestro-v3.0.0.csv"
MAESTRO_DIR = Path(__file__).parent / "maestro-v3.0.0"  # Original MAESTRO dataset directory
OUTPUT_DIR = Path(__file__).parent / "processed"
TRAIN_DIR = OUTPUT_DIR / "train"
TEST_DIR = OUTPUT_DIR / "test"
VALIDATION_DIR = OUTPUT_DIR / "validation"

def organize_midi_files(csv_file, maestro_root, output_root):
    train_dir = output_root / "train"
    test_dir = output_root / "test"
    validation_dir = output_root / "validation"
    
    for dir_path in [train_dir, test_dir, validation_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(csv_file)
    
    print(f"Processing {len(df)} MIDI files from {csv_file}")
    print(f"Source directory: {maestro_root}")
    print(f"Output directory: {output_root}\n")

    stats = {
        'train': {'found': 0, 'missing': 0, 'error': 0},
        'test': {'found': 0, 'missing': 0, 'error': 0},
        'validation': {'found': 0, 'missing': 0, 'error': 0}
    }
    
    grouped = df.groupby('split')
    
    for split_type in ['train', 'test', 'validation']:
        if split_type not in grouped.groups:
            print(f"No {split_type} files in CSV")
            continue
        
        split_df = grouped.get_group(split_type)
        split_output_dir = eval(f"{split_type}_dir")
        
        print(f"\nProcessing {split_type} split ({len(split_df)} files):")
        print("-" * 60)
        
        for idx, row in split_df.iterrows():
            midi_filename = row['midi_filename']
            source_path = maestro_root / midi_filename
            dest_path = split_output_dir / Path(midi_filename).name  # Only filename, no subdirs
            
            try:
                if not source_path.exists():
                    print(f"  [{idx+1:4d}] MISSING: {midi_filename}")
                    stats[split_type]['missing'] += 1
                    continue

                midi = pretty_midi.PrettyMIDI(str(source_path))
                duration = midi.get_end_time()
                num_instruments = len(midi.instruments)

                shutil.copy2(source_path, dest_path)
                
                print(f"  [{idx+1:4d}] OK: {Path(midi_filename).name:60s} ({duration:7.2f}s, {num_instruments} instruments)")
                stats[split_type]['found'] += 1
            
            except FileNotFoundError:
                print(f"  [{idx+1:4d}] MISSING: {midi_filename}")
                stats[split_type]['missing'] += 1
            except Exception as e:
                print(f"  [{idx+1:4d}] ERROR: {midi_filename} - {str(e)[:40]}")
                stats[split_type]['error'] += 1
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    
    total_found = 0
    total_missing = 0
    total_error = 0
    
    for split_type in ['train', 'test', 'validation']:
        s = stats[split_type]
        total = s['found'] + s['missing'] + s['error']
        if total > 0:
            print(f"\n{split_type.upper()}:")
            print(f"  ✓ Successfully copied: {s['found']:4d} files")
            print(f"  ✗ Missing:           {s['missing']:4d} files")
            print(f"  ⚠ Error:             {s['error']:4d} files")
            print(f"  Output: {eval(f'{split_type}_dir')}")
            
            total_found += s['found']
            total_missing += s['missing']
            total_error += s['error']
    
    print("\n" + "=" * 60)
    print(f"TOTAL: {total_found} files copied, {total_missing} missing, {total_error} errors")
    print("=" * 60)
    
    return {
        'output_dir': output_root,
        'train_dir': train_dir,
        'test_dir': test_dir,
        'validation_dir': validation_dir,
        'stats': stats
    }

def load_organized_dataset(output_dir, split='train', max_files=None):
    split_dir = output_dir / split
    
    if not split_dir.exists():
        print(f"Directory not found: {split_dir}")
        return None
    
    midi_files = sorted(list(split_dir.glob("*.midi")) + list(split_dir.glob("*.mid")))
    
    if not midi_files:
        print(f"No MIDI files found in {split_dir}")
        return None
    
    if max_files:
        midi_files = midi_files[:max_files]
    
    midi_data = []
    metadata = []
    failed_files = []
    
    print(f"\nLoading {len(midi_files)} MIDI files from {split} set:")
    print("-" * 60)
    
    for idx, midi_path in enumerate(midi_files):
        try:
            midi = pretty_midi.PrettyMIDI(str(midi_path))
            duration = midi.get_end_time()
            num_instruments = len(midi.instruments)
            
            midi_data.append(midi)
            metadata.append({
                'filename': midi_path.name,
                'path': str(midi_path),
                'duration': duration,
                'instruments': num_instruments
            })
            
            print(f"  [{idx+1:4d}] {midi_path.name:50s} ({duration:7.2f}s, {num_instruments} instruments)")
        
        except Exception as e:
            print(f"  [{idx+1:4d}] ERROR: {midi_path.name} - {str(e)[:40]}")
            failed_files.append(midi_path.name)
    
    print(f"\nLoaded: {len(midi_data)} files")
    print(f"Failed: {len(failed_files)} files")
    
    return {
        'midi_objects': midi_data,
        'metadata': metadata,
        'failed_files': failed_files,
        'split': split,
        'directory': str(split_dir)
    }


if __name__ == "__main__":
    data_dir = Path(__file__).parent
    csv_path = data_dir / CSV_FILE
    maestro_path = data_dir / "maestro-v3.0.0"

    print("STEP 1: ORGANIZING MIDI FILES")
    print("=" * 60)
    result = organize_midi_files(csv_path, maestro_path, OUTPUT_DIR)
    

    print("\n\nSTEP 2: LOADING ORGANIZED DATASET")
    print("=" * 60)

    train_data = load_organized_dataset(OUTPUT_DIR, split='train', max_files=5)
    
    if train_data and train_data['metadata']:
        print("\nTRAIN SET STATISTICS:")
        print("-" * 60)
        meta_df = pd.DataFrame(train_data['metadata'])
        print(f"Total files: {len(train_data['midi_objects'])}")
        print(f"Average duration: {meta_df['duration'].mean():.2f}s")
        print(f"Total duration: {meta_df['duration'].sum():.2f}s")
        print(f"Average instruments: {meta_df['instruments'].mean():.2f}")