import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim*2),
            nn.ReLU(),
            nn.Linear(in_dim*2, out_dim)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class StyleMapping(nn.Module):
    def __init__(self, style_dim,  style_heads=3, mlp_dim=512):
        super(StyleMapping, self).__init__()
        self.styles = nn.ModuleList()
        for i in range(style_heads):
            self.styles.append(MLP(style_dim, mlp_dim))

    def forward(self, latent):
        #  latent BxD -> style BxLxD
        res = []
        # print(latent.shape, 'voice latent')
        for m in self.styles:
            res.append(m(latent.squeeze()).unsqueeze(1))

        res = torch.cat(res, dim=1)
        return res


class VoiceStyleNet(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(VoiceStyleNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(input_channel, channels[0], 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels[0], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels[0], channels[1], 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels[1], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels[1], channels[2], 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels[2], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels[2], channels[3], 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels[3], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels[3], output_channel, 3, 2, 1, bias=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.style_mapping = StyleMapping(64, 64, 512)

    def forward(self, x):
        x = self.model(x)
        out = self.style_mapping(x)
        return out