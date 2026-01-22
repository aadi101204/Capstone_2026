import torch
import torch.nn as nn

class LSTMGenerator(nn.Module):
    def __init__(self, noise_dim=64, hidden_dim=128, seq_length=192):
        super(LSTMGenerator, self).__init__()
        self.seq_length = seq_length
        self.fc = nn.Linear(noise_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.bn = nn.BatchNorm1d(seq_length)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward_branch(self, z):
        x = self.fc(z).unsqueeze(1).repeat(1, self.seq_length, 1)
        lstm_out, _ = self.lstm(x)
        out = self.fc_out(lstm_out)
        return out.squeeze(-1)

    def forward(self, z):
        out1 = self.forward_branch(z)
        out2 = self.forward_branch(z)
        out3 = self.forward_branch(z)
        out4 = self.forward_branch(z)
        merged = (out1 + out2 + out3 + out4) / 4.0
        merged = torch.sigmoid(self.bn(merged))
        return merged

class LSTMCritic(nn.Module):
    def __init__(self, hidden_dim=128):
        super(LSTMCritic, self).__init__()
        self.lstm = nn.LSTM(1, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, seq):
        lstm_out, _ = self.lstm(seq)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)
