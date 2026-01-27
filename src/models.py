import torch
import torch.nn as nn

class LSTMGeneratorBranch(nn.Module):
    def __init__(self, noise_slice_dim, hidden_dim, seq_length):
        super(LSTMGeneratorBranch, self).__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        
        # SIREN-style Sine Initialization for high-frequency chaotic mapping
        self.init_h = nn.Linear(noise_slice_dim, hidden_dim)
        self.init_c = nn.Linear(noise_slice_dim, hidden_dim)
        
        # LSTM layer (Using 3 layers for more complex internal state dynamics)
        self.lstm = nn.LSTM(1, hidden_dim, num_layers=3, batch_first=True, dropout=0.1)
        
        # ELU is better than ReLU/LeakyReLU for preserving fluctuations in chaos
        self.fc_delta = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, z_slice):
        batch_size = z_slice.size(0)
        
        # Sine-based seeding: Maps static noise to a high-frequency initial state
        # Multiplying by 30 is a standard SIREN hyperparameter for capturing fine details
        h0 = torch.sin(30 * self.init_h(z_slice)).unsqueeze(0).repeat(3, 1, 1).contiguous()
        c0 = torch.sin(30 * self.init_c(z_slice)).unsqueeze(0).repeat(3, 1, 1).contiguous()
        
        # Integrate dummy zeros to evolve the state
        dummy_input = z_slice.new_zeros(batch_size, self.seq_length, 1)
        lstm_out, _ = self.lstm(dummy_input, (h0, c0))
        
        # Residual Delta Output scaled by 0.1 for stability
        deltas = self.fc_delta(lstm_out).squeeze(-1) * 0.1
        
        # Integration (Step-by-step accumulation)
        seq = torch.cumsum(deltas, dim=1)
        
        return seq

class LSTMGenerator(nn.Module):
    def __init__(self, noise_dim=64, hidden_dim=128, seq_length=192):
        super(LSTMGenerator, self).__init__()
        self.seq_length = seq_length
        self.noise_dim = noise_dim
        
        slice_dim = noise_dim // 4
        
        self.branch1 = LSTMGeneratorBranch(slice_dim, hidden_dim, seq_length)
        self.branch2 = LSTMGeneratorBranch(slice_dim, hidden_dim, seq_length)
        self.branch3 = LSTMGeneratorBranch(slice_dim, hidden_dim, seq_length)
        self.branch4 = LSTMGeneratorBranch(slice_dim, hidden_dim, seq_length)
        
        # REMOVED Coupling (Conv1d) and BatchNorm here as they smooth out the chaos.
        # Direct stacking preserve maximum entropy per channel.

    def forward(self, z):
        s = self.noise_dim // 4
        out1 = self.branch1(z[:, 0:s])
        out2 = self.branch2(z[:, s:2*s])
        out3 = self.branch3(z[:, 2*s:3*s])
        out4 = self.branch4(z[:, 3*s:4*s])
        
        # Stack into (Batch, 4, Seq_Length)
        stacked = torch.stack([out1, out2, out3, out4], dim=1)
        
        # Global scaling to [0, 1] for images/NIST compatibility
        # Using Sigmoid directly on the integrated signal
        return torch.sigmoid(stacked)

class LSTMCritic(nn.Module):
    def __init__(self, hidden_dim=128):
        super(LSTMCritic, self).__init__()
        # 2 layers are sufficient and much faster for the backward pass
        self.lstm_raw = nn.LSTM(4, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.lstm_diff = nn.LSTM(4, hidden_dim, num_layers=1, batch_first=True)
        
        # Spectral Normalization on the FC layers resolves the hang issues 
        # and provides WGAN stability without needing the expensive Gradient Penalty.
        self.fc = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(hidden_dim * 4, hidden_dim)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, 1))
        )

    def forward(self, seq):
        # seq shape: (batch_size, 4, seq_length)
        seq = seq.permute(0, 2, 1)
        
        # 1. Raw Stream
        raw_out, _ = self.lstm_raw(seq)
        avg_raw = torch.mean(raw_out, dim=1)
        max_raw, _ = torch.max(raw_out, dim=1)
        
        # 2. Differential Stream (Penalizes smoothness)
        diff = seq[:, 1:, :] - seq[:, :-1, :]
        diff_out, _ = self.lstm_diff(diff)
        avg_diff = torch.mean(diff_out, dim=1)
        max_diff, _ = torch.max(diff_out, dim=1)
        
        combined = torch.cat([avg_raw, max_raw, avg_diff, max_diff], dim=1)
        return self.fc(combined)
