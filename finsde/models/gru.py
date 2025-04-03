import torch
import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.linear = nn.Linear(
            in_features=hidden_size * n_layers * 2,
            out_features=output_size,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_size)
        h_n = self.gru(x)[1]
        # h_n shape: (num_layers * num_directions, batch_size, hidden_size)
        h_n = h_n.permute(1, 0, 2).flatten(start_dim=1)
        # h_n shape: (batch_size, num_layers * num_directions * hidden_size)
        out = self.linear(h_n)
        # out shape: (batch_size, output_size)
        return out