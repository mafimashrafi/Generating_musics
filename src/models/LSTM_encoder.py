import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import pickle
import numpy as np


class LSTMEncoder(nn.Module):
    def __init__(self, input_size=88, hidden_size=256, num_layers=2, 
                latent_dim=64, dropout=0.2, bidirectional=True):
        super(LSTMEncoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.bidirectional = bidirectional

        self.lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        self.fc1 = nn.Linear(self.lstm_output_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, latent_dim)

        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(latent_dim)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):

        batch_size = x.size(0)

        lstm_out, (hidden, cell) = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        
        x = self.fc1(last_out)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)

        latent = x
        
        return latent, (hidden, cell)
    
    def encode_sequence(self, x):
        latent, _ = self.forward(x)
        return latent


class LSTMDecoder(nn.Module):

    def __init__(self, latent_dim=64, hidden_size=256, num_layers=2,
                output_size=88, seq_len=64, dropout=0.2):

        super(LSTMDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.seq_len = seq_len

        self.latent_to_hidden = nn.Linear(latent_dim, hidden_size * num_layers)
        self.latent_to_cell = nn.Linear(latent_dim, hidden_size * num_layers)

        self.lstm = nn.LSTM(
            input_size=output_size,  
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc_out = nn.Linear(hidden_size, output_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, latent, teacher_forcing_ratio=0.0, target=None):
        batch_size = latent.size(0)
        device = latent.device

        hidden = self.latent_to_hidden(latent).view(
            self.num_layers, batch_size, self.hidden_size
        )
        cell = self.latent_to_cell(latent).view(
            self.num_layers, batch_size, self.hidden_size
        )

        if target is not None and torch.rand(1).item() > teacher_forcing_ratio:
            input_t = target[:, 0, :]  # Use first timestep of target
        else:
            input_t = torch.zeros(batch_size, self.output_size, device=device)
        
        outputs = []
        
        for t in range(self.seq_len):
            lstm_out, (hidden, cell) = self.lstm(
                input_t.unsqueeze(1), (hidden, cell)
            )

            output = self.fc_out(lstm_out.squeeze(1))
            output = torch.sigmoid(output)  # Binary piano roll
            outputs.append(output)

            if target is not None and torch.rand(1).item() > teacher_forcing_ratio:
                input_t = target[:, t, :]
            else:
                input_t = output
        
        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, output_size)
        return outputs


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=88, hidden_size=256, num_layers=2,
                latent_dim=64, seq_len=64, dropout=0.2, bidirectional=True):
        super(LSTMAutoencoder, self).__init__()
        
        self.encoder = LSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            latent_dim=latent_dim,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        self.decoder = LSTMDecoder(
            latent_dim=latent_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=input_size,
            seq_len=seq_len,
            dropout=dropout
        )
        
    def forward(self, x, teacher_forcing_ratio=0.0):
        latent, _ = self.encoder(x)
        reconstructed = self.decoder(latent, teacher_forcing_ratio, target=x)
        return reconstructed, latent


def load_preprocessed_data(data_dir='data/preprocessed_output/sequences/train', 
                        max_files=None):

    data_path = Path(data_dir)
    pkl_files = sorted(list(data_path.glob('*.pkl')))
    
    if max_files:
        pkl_files = pkl_files[:max_files]
    
    print(f"Loading {len(pkl_files)} files from {data_dir}...")
    
    all_segments = []
    for i, pkl_file in enumerate(pkl_files):
        with open(pkl_file, 'rb') as f:
            segments = pickle.load(f)
            all_segments.append(segments)
        
        if (i + 1) % 100 == 0:
            print(f"  Loaded {i+1}/{len(pkl_files)} files...")
    
    # Concatenate all segments
    dataset = np.concatenate(all_segments, axis=0)
    print(f"✓ Loaded dataset shape: {dataset.shape}")
    
    return dataset


class MIDIDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir='data/preprocessed_output/sequences/train', 
                max_files=None):
        self.data = load_preprocessed_data(data_dir, max_files)
        self.data = torch.FloatTensor(self.data)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def test_model():
    print("Testing LSTM Encoder...")
    batch_size = 4
    seq_len = 64
    n_pitches = 88
    
    dummy_input = torch.randn(batch_size, seq_len, n_pitches)
    
    # Test encoder
    encoder = LSTMEncoder(input_size=88, latent_dim=64)
    latent, (hidden, cell) = encoder(dummy_input)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Latent shape: {latent.shape}")
    
    # Test decoder
    decoder = LSTMDecoder(latent_dim=64, output_size=88, seq_len=64)
    reconstructed = decoder(latent)
    print(f"  Reconstructed shape: {reconstructed.shape}")
    
    # Test autoencoder
    autoencoder = LSTMAutoencoder(input_size=88, latent_dim=64)
    recon, latent_out = autoencoder(dummy_input)
    print(f"  Autoencoder output shape: {recon.shape}")
    print(f"  Latent shape: {latent_out.shape}")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_model()