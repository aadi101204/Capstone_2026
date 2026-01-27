import torch
import torch.nn as nn

class LSTMGeneratorBranch(nn.Module):
    def __init__(self, noise_slice_dim, hidden_dim, seq_length):
        super(LSTMGeneratorBranch, self).__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        
        # Initial state generator: maps noise to initial h and c
        # Each branch gets its own seed generator
        self.init_h = nn.Linear(noise_slice_dim, hidden_dim)
        self.init_c = nn.Linear(noise_slice_dim, hidden_dim)
        
        # LSTM layer (deeper for complexity)
        # Note: We don't feed noise at every step, just the initial state
        self.lstm = nn.LSTM(1, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh() # Tanh helps centered chaotic trajectories
        )
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, z_slice):
        batch_size = z_slice.size(0)
        
        # Initialize h and c using the noise "seed"
        h0 = self.leaky_relu(self.init_h(z_slice)).unsqueeze(0).repeat(2, 1, 1).contiguous()
        c0 = self.leaky_relu(self.init_c(z_slice)).unsqueeze(0).repeat(2, 1, 1).contiguous()
        
        # Using .new_zeros() is faster and stays on the same device as z_slice
        dummy_input = z_slice.new_zeros(batch_size, self.seq_length, 1)
        
        lstm_out, _ = self.lstm(dummy_input, (h0, c0))
        
        out = self.fc_out(lstm_out)
        return out.squeeze(-1)

class LSTMGenerator(nn.Module):
    def __init__(self, noise_dim=64, hidden_dim=128, seq_length=192):
        super(LSTMGenerator, self).__init__()
        self.seq_length = seq_length
        self.noise_dim = noise_dim
        
        # Split noise_dim into 4 slices for 4-node independence
        slice_dim = noise_dim // 4
        
        self.branch1 = LSTMGeneratorBranch(slice_dim, hidden_dim, seq_length)
        self.branch2 = LSTMGeneratorBranch(slice_dim, hidden_dim, seq_length)
        self.branch3 = LSTMGeneratorBranch(slice_dim, hidden_dim, seq_length)
        self.branch4 = LSTMGeneratorBranch(slice_dim, hidden_dim, seq_length)
        
        # Step 1: Cross-Node Interaction (Coupled Dynamics)
        # 1D Convolution to mix dynamics between nodes across time
        self.coupling = nn.Conv1d(4, 4, kernel_size=3, padding=1, padding_mode='circular')
        
        self.bn = nn.BatchNorm1d(4)

    def forward(self, z):
        # Slice noise for each branch
        s = self.noise_dim // 4
        out1 = self.branch1(z[:, 0:s])
        out2 = self.branch2(z[:, s:2*s])
        out3 = self.branch3(z[:, 2*s:3*s])
        out4 = self.branch4(z[:, 3*s:4*s])
        
        # Stack into (Batch, 4, Seq_Length)
        stacked = torch.stack([out1, out2, out3, out4], dim=1)
        
        # Apply Cross-Node Coupling to increase complexity/randomness
        coupled = self.coupling(stacked)
        
        # Batch normalization across the 4 nodes
        normalized = self.bn(coupled)
        
        # Rescale to [0, 1]
        return torch.sigmoid(normalized)

class LSTMCritic(nn.Module):
    def __init__(self, hidden_dim=128):
        super(LSTMCritic, self).__init__()
        # input_size=4 because we evaluate 4 chaotic nodes simultaneously
        self.lstm_raw = nn.LSTM(4, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.lstm_diff = nn.LSTM(4, hidden_dim, num_layers=1, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, seq):
        # seq shape: (batch_size, 4, seq_length)
        # Convert to (batch_size, seq_length, 4) for LSTM processing
        seq = seq.permute(0, 2, 1)
        
        # 1. Raw features
        raw_out, _ = self.lstm_raw(seq)
        avg_raw = torch.mean(raw_out, dim=1)
        max_raw, _ = torch.max(raw_out, dim=1)
        
        # 2. Differential features (Chaos sensitivity)
        diff = seq[:, 1:, :] - seq[:, :-1, :]
        diff_out, _ = self.lstm_diff(diff)
        avg_diff = torch.mean(diff_out, dim=1)
        max_diff, _ = torch.max(diff_out, dim=1)
        
        # Combine everything
        combined = torch.cat([avg_raw, max_raw, avg_diff, max_diff], dim=1)
        
        return self.fc(combined)
