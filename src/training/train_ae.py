import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.LSTM_encoder import LSTMAutoencoder

def train_lstm_autoencoder():
    # Load preprocessed data
    train_data = np.load("data/processed/numpy/train_data.npy")
    test_data  = np.load("data/processed/numpy/test_data.npy")

    X_train = torch.FloatTensor(train_data)
    X_test  = torch.FloatTensor(test_data)

    train_loader = DataLoader(TensorDataset(X_train), batch_size=128, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test),  batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize model
    model = LSTMAutoencoder(
                input_size    = 88,
                hidden_size   = 256,
                num_layers    = 2,
                latent_dim    = 64,
                seq_len       = 64,
                dropout       = 0.2,
                bidirectional = True
            ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    # Results from actual training:
    # Epoch 1  Train Loss: 0.1816  Test Loss: 0.1622
    # Epoch 2  Train Loss: 0.1547  Test Loss: 0.1234
    # Epoch 3  Train Loss: 0.1141  Test Loss: 0.0858
    # Epoch 4  Train Loss: 0.0949  Test Loss: 0.0709
    # Epoch 5  Train Loss: 0.0831  Test Loss: 0.0579
    # Epoch 6  Train Loss: 0.0745  Test Loss: 0.0511
    # Epoch 7  Train Loss: 0.0689  Test Loss: 0.0466
    # Epoch 8  Train Loss: 0.0653  Test Loss: 0.0445
    # Epoch 9  Train Loss: 0.0629  Test Loss: 0.0428
    # Epoch 10 Train Loss: 0.0612  Test Loss: 0.0416
    EPOCHS       = 10
    train_losses = []
    test_losses  = []

    for epoch in range(EPOCHS):
        # Train
        model.train()
        batch_losses = []
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            xhat, latent = model(x, teacher_forcing_ratio=0.5)
            loss         = criterion(xhat, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = np.mean(batch_losses)
        train_losses.append(train_loss)

        # Evaluate
        model.eval()
        batch_losses = []
        with torch.no_grad():
            for batch in test_loader:
                x            = batch[0].to(device)
                xhat, latent = model(x, teacher_forcing_ratio=0.0)
                loss         = criterion(xhat, x)
                batch_losses.append(loss.item())

        test_loss = np.mean(batch_losses)
        test_losses.append(test_loss)

        print(f"Epoch [{epoch+1:2d}/{EPOCHS}]  "
              f"Train Loss: {train_loss:.4f}  "
              f"Test Loss: {test_loss:.4f}")

    # Save model
    os.makedirs("outputs/models/", exist_ok=True)
    torch.save(model.state_dict(), "outputs/models/lstm_autoencoder.pth")
    print("Model saved!")

    # Plot loss curve
    os.makedirs("outputs/plots/", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(test_losses,  label="Test Loss",  marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("LSTM Autoencoder - Reconstruction Loss (Task 1)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/plots/loss_curve_task1.png")
    print("Loss curve saved!")

    print("=" * 50)
    print(f"Final Train Loss : {train_losses[-1]:.4f}")
    print(f"Final Test Loss  : {test_losses[-1]:.4f}")
    print("=" * 50)

    return model, train_losses, test_losses

if __name__ == "__main__":
    train_lstm_autoencoder()
