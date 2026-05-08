import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(logits, targets, gamma=2.0, pos_weight=20.0):
    """
    Focal Loss for sparse piano roll data.
    Piano rolls are 97-98% zeros — plain BCE fails because the model
    learns to predict zero everywhere. Focal Loss fixes this by:
    - pos_weight=20: active notes count 20x more than silence
    - gamma=2.0: suppresses gradient from easy (correctly silent) cells
    """
    bce = F.binary_cross_entropy_with_logits(
        logits, targets,
        pos_weight=torch.tensor(pos_weight).to(logits.device),
        reduction="none"
    )
    probs = torch.sigmoid(logits)
    pt    = torch.where(targets == 1, probs, 1 - probs)
    return ((1 - pt) ** gamma * bce).mean()


class MusicVAE(nn.Module):
    """
    VAE extending Task 1 LSTM Autoencoder.
    Encoder outputs mu and logvar instead of single z.
    Reparameterization trick enables backprop through sampling.
    KL annealing prevents posterior collapse.
    """
    def __init__(self, input_size=88, hidden_size=256, latent_dim=64, num_layers=2):
        super(MusicVAE, self).__init__()
        self.num_layers  = num_layers
        self.hidden_size = hidden_size
        self.latent_dim  = latent_dim

        # Bidirectional LSTM encoder
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                    batch_first=True, dropout=0.3, bidirectional=True)
        self.fc_mu     = nn.Linear(hidden_size * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size * 2, latent_dim)

        # Autoregressive LSTM decoder
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_size * num_layers)
        self.latent_to_cell   = nn.Linear(latent_dim, hidden_size * num_layers)
        self.decoder_lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                    batch_first=True, dropout=0.3)
        self.fc_out = nn.Linear(hidden_size, input_size)  # raw logits — no sigmoid

    def encode(self, x):
        _, (hidden, _) = self.encoder_lstm(x)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        mu     = self.fc_mu(hidden)
        logvar = torch.clamp(self.fc_logvar(hidden), min=-4.0, max=4.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode_autoregressive(self, z, seq_len=64):
        batch_size = z.size(0)
        hidden = self.latent_to_hidden(z).view(self.num_layers, batch_size, self.hidden_size)
        cell   = self.latent_to_cell(z).view(self.num_layers, batch_size, self.hidden_size)
        x      = torch.zeros(batch_size, 1, 88).to(z.device)
        outputs = []
        for _ in range(seq_len):
            out, (hidden, cell) = self.decoder_lstm(x, (hidden, cell))
            prob = torch.sigmoid(self.fc_out(out))
            outputs.append(prob)
            x = prob
        return torch.cat(outputs, dim=1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        hidden = self.latent_to_hidden(z).view(self.num_layers, x.size(0), self.hidden_size)
        cell   = self.latent_to_cell(z).view(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.decoder_lstm(x, (hidden, cell))
        return self.fc_out(out), mu, logvar  # raw logits


def vae_loss(logits, x, mu, logvar, beta=1.0):
    recon = focal_loss(logits, x)
    kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl
