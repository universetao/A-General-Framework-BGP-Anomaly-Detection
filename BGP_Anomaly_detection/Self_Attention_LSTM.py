import torch
import torch.nn as nn
import torch.nn.functional as F
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights

class SA_LSTM(nn.Module):
    def __init__(self,WINDOW_SIZE,INPUT_SIZE,Hidden_SIZE,LSTM_layer_NUM):
        super(SA_LSTM, self).__init__()
        self.WINDOW_SIZE = WINDOW_SIZE
        self.INPUT_SIZE = INPUT_SIZE
        self.Hidden_SIZE = Hidden_SIZE
        self.LSTM_layer_NUM = LSTM_layer_NUM
        # self.LN1=nn.LayerNorm(INPUT_SIZE)
        self.BN = nn.BatchNorm1d(self.WINDOW_SIZE)
        self.lstm = nn.LSTM(input_size=INPUT_SIZE,
                            hidden_size=Hidden_SIZE,
                            num_layers=LSTM_layer_NUM,
                            batch_first=True,
                            # dropout=0.5
                            )
        self.attention = SelfAttention(Hidden_SIZE)
        self.out = nn.Sequential(nn.Linear(Hidden_SIZE, 2), nn.Softmax())

    def forward(self, x):
        x=self.BN(x)
        r_out, hidden = self.lstm(x, None)  # x(batch,time_step,input_size)
        r_out, attn_weights = self.attention(r_out)
        out = self.out(r_out)
        # outputs = self.fc(r_out.view(BATCH_SIZE, -1))
        return out ,torch.mean(attn_weights,dim=-2)