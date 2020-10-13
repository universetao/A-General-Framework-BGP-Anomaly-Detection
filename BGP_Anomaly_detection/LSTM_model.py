import torch
import torch.nn as nn
class LSTM(nn.Module):
    def __init__(self,WINDOW_SIZE,INPUT_SIZE,Hidden_SIZE,LSTM_layer_NUM):
        super(LSTM, self).__init__()
        # self.LN1=nn.LayerNorm(INPUT_SIZE)
        self.WINDOW_SIZE=WINDOW_SIZE
        self.INPUT_SIZE=INPUT_SIZE
        self.Hidden_SIZE=Hidden_SIZE
        self.LSTM_layer_NUM=LSTM_layer_NUM
        self.BN = nn.BatchNorm1d(self.WINDOW_SIZE)
        self.lstm = nn.LSTM(input_size=self.INPUT_SIZE,
                             hidden_size=self.Hidden_SIZE,
                             num_layers=self.LSTM_layer_NUM,
                             batch_first=True,
                             # dropout=0.5
                             )

        self.out = nn.Sequential(nn.Linear(self.Hidden_SIZE, 2), nn.Softmax())

    def forward(self, x):
        x = self.BN(x)
        r_out, (h_n1, h_c1) = self.lstm(x, None)  # x(batch,time_step,input_size)
        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out
