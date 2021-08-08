import torch
import torch.nn as nn
import torch.nn.functional as F
from models.stylegan2 import Generator



class MLP(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MLP, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.ReLU(True),
            nn.Linear(out_channel, out_channel),
            nn.LeakyReLU(0.01),
            nn.Linear(out_channel, out_channel)
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class VoiceEmbedNet(nn.Module):
    def __init__(self, stylegan: Generator):
        super(VoiceEmbedNet, self).__init__()

        self.g = stylegan
        for p in self.g.parameters():
            p.requires_grad = False

        self.model = nn.Sequential(
            nn.Conv1d(64, 512, 3, 1, 1, bias=False),
            nn.BatchNorm1d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 3, 2, 1, bias=False),
            nn.BatchNorm1d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm1d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 3, 2, 1, bias=False),
            nn.BatchNorm1d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 3, 2, 1, bias=True),
        )

        self.z_mapping = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 512,),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 512)
        )

        self.mapping = nn.ModuleList()
        for i in range(4):
            x = MLP(512, 512)
            self.mapping.append(x)

    @torch.no_grad()
    def mean_latent(self, x):
        """
        Anchor latent feature for one's.
        :param x: BxD
        :return: 1xD
        """

        f = self.model(x)
        z = self.z_mapping(f)
        out = self.g.style(z).mean(0, keepdim=True)
        return out

    def forward(self, x, mean_styles=None):

        feature = self.model(x)
        # print(feature.shape)
        z = self.z_mapping(feature)

        latent_codes = [m(z) for m in self.mapping]
        assert len(latent_codes) == 4

        if mean_styles is not None:
            styles = [0.6*self.g.style(z)+0.4*mean_styles]
            for l in latent_codes:
                styles.append(0.6*self.g.style(l)+0.4*mean_styles)
        else:
            styles = [self.g.style(z)]
            for l in latent_codes:
                styles.append(self.g.style(l))

        batch_size = x.shape[0]
        out = self.g.input(styles[0])
        out = self.g.conv1(out, styles[0], noise=torch.rand(batch_size, 512, 4, 4, device=x.device)*styles[0][:, :, None, None])
        skip = self.g.to_rgb1(out, styles[0])

        i = 1
        size = 8
        for conv1, conv2, style_feature, to_rgb in zip(
                self.g.convs[::2], self.g.convs[1::2], styles, self.g.to_rgbs
        ):
            out = conv1(out, style_feature, noise=torch.rand(batch_size, 512, size, size, device=x.device)*style_feature[:, :, None, None])
            out = conv2(out, style_feature, noise=torch.rand(batch_size, 512, size, size, device=x.device)*style_feature[:, :, None, None])
            skip = to_rgb(out, style_feature, skip)
            size *= 2
            i += 2
        image = skip
        return image
