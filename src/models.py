import torch
import torch.nn as nn

class LSTMGeneratorBranch(nn.Module):
    def __init__(self, noise_dim, hidden_dim, seq_length):
        super(LSTMGeneratorBranch, self).__init__()
        self.seq_length = seq_length
        self.fc = nn.Linear(noise_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, z):
        # Initial projection and expansion to sequence length
        x = self.leaky_relu(self.fc(z)).unsqueeze(1).repeat(1, self.seq_length, 1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Skip connection: add projected input to LSTM output if dimensions match
        # Here we add the initial x to lstm_out
        res = lstm_out + x
        
        out = self.fc_out(res)
        return out.squeeze(-1)

class LSTMGenerator(nn.Module):
    def __init__(self, noise_dim=64, hidden_dim=128, seq_length=192):
        super(LSTMGenerator, self).__init__()
        self.seq_length = seq_length
        
        # 4 Independent branches to preserve entropy
        self.branch1 = LSTMGeneratorBranch(noise_dim, hidden_dim, seq_length)
        self.branch2 = LSTMGeneratorBranch(noise_dim, hidden_dim, seq_length)
        self.branch3 = LSTMGeneratorBranch(noise_dim, hidden_dim, seq_length)
        self.branch4 = LSTMGeneratorBranch(noise_dim, hidden_dim, seq_length)
        
        self.bn = nn.BatchNorm1d(seq_length)

    def forward(self, z):
        out1 = self.branch1(z)
        out2 = self.branch2(z)
        out3 = self.branch3(z)
        out4 = self.branch4(z)
        
        # Maintain the 4-node averaging requirement
        merged = (out1 + out2 + out3 + out4) / 4.0
        
        # Normalization and output mapping
        merged = torch.sigmoid(self.bn(merged))
        return merged

class LSTMCritic(nn.Module):
    def __init__(self, hidden_dim=128):
        super(LSTMCritic, self).__init__()
        # Deeper LSTM for critic
        self.lstm = nn.LSTM(1, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        
        # Multi-scale pooling to capture global randomness features
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, seq):
        # seq shape: (batch_size, seq_length, 1)
        lstm_out, _ = self.lstm(seq)
        
        # Global Average Pooling
        avg_pool = torch.mean(lstm_out, dim=1)
        # Global Max Pooling
        max_pool, _ = torch.max(lstm_out, dim=1)
        
        # Concatenate pooling results (captures both trend and volatility)
        combined = torch.cat([avg_pool, max_pool], dim=1)
        
        return self.fc(combined)
