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