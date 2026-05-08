import math
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_TOKEN   = 128   
BOS_TOKEN   = 129   # beginning-of-sequence
EOS_TOKEN   = 130   # end-of-sequence
EMPTY_TOKEN = 131   
SEP_TOKEN   = 132   # end of one timestep's pitch list
VOCAB_SIZE  = 133   


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MusicTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int   = VOCAB_SIZE,
        d_model:    int   = 256,
        nhead:      int   = 8,
        num_layers: int   = 6,
        dim_ff:     int   = 1024,
        max_len:    int   = 4096,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_ff,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,  # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out      = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _causal_mask(sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        x:        torch.Tensor,            # (B, T)
        pad_mask: torch.Tensor | None = None,  # (B, T) True = pad
    ) -> torch.Tensor:                     # (B, T, vocab_size)
        T      = x.size(1)
        causal = self._causal_mask(T, x.device)
        emb    = self.pos_enc(self.embedding(x) * math.sqrt(self.d_model))
        out    = self.transformer(emb, mask=causal, src_key_padding_mask=pad_mask)
        return self.fc_out(out)

    @torch.no_grad()
    def generate(
        self,
        prompt:      torch.Tensor,    # (1, T_prompt)
        max_new:     int   = 512,
        temperature: float = 1.0,
        top_k:       int   = 50,
    ) -> torch.Tensor:                # (1, T_prompt + generated)
        self.eval()
        seq = prompt.clone()

        for _ in range(max_new):
            logits = self(seq)[:, -1, :] / temperature  # (1, V)

            if top_k > 0:
                vals, _ = logits.topk(top_k, dim=-1)
                logits  = logits.masked_fill(logits < vals[:, -1:], float("-inf"))

            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            seq      = torch.cat([seq, next_tok], dim=1)

            if next_tok.item() == EOS_TOKEN:
                break

        return seq