import torch
import torch.nn as nn

class S2CG(nn.Module):
    def __init__(self, length=100, hidden_dim=(256,32,64), cutoff=25, threshold=0):
        super().__init__()
        self.proj1 = nn.Sequential(
            nn.Linear(length, hidden_dim[0]),
            nn.GELU(),
        )
        self.proj2 = nn.Sequential(
            nn.Linear(length//2+1, hidden_dim[1]),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(1, hidden_dim[2], kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim[2], 1, kernel_size=3, stride=1, padding=1),
        )
        self.threshold = threshold
        self.cutoff = cutoff
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, ts):
        b, n, t = ts.shape
        x_f = torch.fft.rfft(ts, dim=-1)
        x_f[:, :, 0] = 0
        f_t = torch.abs(x_f)
        mask = torch.zeros_like(x_f, dtype=torch.bool)
        _, indices = torch.topk(torch.abs(x_f), self.cutoff, dim=-1)
        mask.scatter_(2, indices, True)

        result = torch.zeros_like(x_f)
        result[mask] = x_f[mask]
        ts = torch.fft.irfft(result, dim=-1, n=t)
        x, s = self.proj1(ts) , self.proj2(f_t)
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        s = torch.nn.functional.normalize(s, p=2, dim=-1)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        s1, s2 = torch.chunk(s, 2, dim=-1)
        spatial_sim = x1@x2.transpose(-1,-2)
        spectral_sim = s1@s2.transpose(-1,-2)
        sim_mat = spectral_sim + spatial_sim # BxNxN
        sim_mat = self.proj(sim_mat.unsqueeze(1)).squeeze(1)

        sim_mat = sim_mat.permute(0, 2, 1) + sim_mat
        diagonal_matrix = torch.eye(n).unsqueeze(0).repeat(b, 1, 1).to(ts.device)
        sim_mat = sim_mat * (1 - diagonal_matrix)
        sim_mat[sim_mat < self.threshold] = 0

        return sim_mat, spatial_sim, spectral_sim