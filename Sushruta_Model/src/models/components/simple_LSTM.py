from torch import nn


class SimpleLSTM(nn.Module):
    def __init__ (
        self,
        input_size: int = 8,
        hidden_size: int = 12,
        num_layers: int = 1,
    ):
        super().__init__()
    
        self.model = nn.Sequential(
            nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True),
        )

    def forward (self, x): 
        return self.model(x)