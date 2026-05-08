import json
import random
from pathlib import Path
from typing import List, Sequence

import torch
from torch.utils.data import Dataset

PAD_TOKEN   = 128
BOS_TOKEN   = 129
EOS_TOKEN   = 130
EMPTY_TOKEN = 131
SEP_TOKEN   = 132
VOCAB_SIZE  = 133

def pianoroll_to_tokens(timesteps: List[List[int]]) -> List[int]:
    """List[List[pitch]] → flat token sequence with BOS/EOS."""
    tokens: List[int] = [BOS_TOKEN]
    for pitches in timesteps:
        if not pitches:
            tokens.append(EMPTY_TOKEN)
        else:
            for p in sorted(pitches):
                tokens.append(int(p))
            tokens.append(SEP_TOKEN)
    tokens.append(EOS_TOKEN)
    return tokens

def tokens_to_pianoroll(tokens: List[int]) -> List[List[int]]:
    timesteps: List[List[int]] = []
    current:   List[int]       = []

    for tok in tokens:
        if tok == BOS_TOKEN:
            continue
        if tok == EOS_TOKEN:
            if current:
                timesteps.append(sorted(current))
            break
        if tok == PAD_TOKEN:
            continue
        if tok == EMPTY_TOKEN:
            timesteps.append([])
            current = []
        elif tok == SEP_TOKEN:
            timesteps.append(sorted(current))
            current = []
        elif 0 <= tok <= 127:
            current.append(tok)

    return timesteps

class MidiTokenDataset(Dataset):
    def __init__(
        self,
        json_paths: Sequence[Path | str],
        seq_len:    int  = 512,
        stride:     int  = 256,
        augment:    bool = False,
    ):
        self.seq_len = seq_len
        self.augment = augment
        self.sequences: List[List[int]] = []

        for path in json_paths:
            with open(path) as f:
                raw: List[List[int]] = json.load(f)
            tokens = pianoroll_to_tokens(raw)
            for start in range(0, max(1, len(tokens) - seq_len), stride):
                chunk = tokens[start : start + seq_len + 1]
                if len(chunk) >= 2:
                    self.sequences.append(chunk)

        if not self.sequences:
            raise ValueError("No sequences found — check your JSON paths.")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        tokens = list(self.sequences[idx])

        if self.augment:
            shift  = random.randint(-6, 6)
            tokens = [
                min(127, max(0, t + shift)) if 0 <= t <= 127 else t
                for t in tokens
            ]

        target_len = self.seq_len + 1
        if len(tokens) < target_len:
            tokens += [PAD_TOKEN] * (target_len - len(tokens))
        tokens = tokens[:target_len]

        src      = torch.tensor(tokens[:-1], dtype=torch.long)
        tgt      = torch.tensor(tokens[1:],  dtype=torch.long)
        pad_mask = (src == PAD_TOKEN)
        return src, tgt, pad_mask

def collate_fn(batch):
    srcs, tgts, masks = zip(*batch)
    return torch.stack(srcs), torch.stack(tgts), torch.stack(masks)