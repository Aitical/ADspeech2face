import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from einops import rearrange, reduce, repeat


class GlobalContext(nn.Module):
    def __init__(
        self,
        *,
        chan_in,
        chan_out
    ):
        super().__init__()
        self.to_k = nn.Conv2d(chan_in, 1, 1)
        chan_intermediate = max(3, chan_out // 2)

        self.net = nn.Sequential(
            nn.Conv2d(chan_in, chan_intermediate, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(chan_intermediate, chan_out, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        context = self.to_k(x)
        context = context.flatten(2).softmax(dim = -1)
        out = einsum('b i n, b c n -> b c i', context, x.flatten(2))
        out = out.unsqueeze(-1)
        return self.net(out)




class Upsample(nn.Module):
    def __init__(self, chan_in, chan_out, scale_factor=2):
        super(Upsample, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor),
            # Blur(),
            nn.Conv2d(chan_in, chan_out * 2, 3, padding=1),
            nn.BatchNorm2d(chan_out * 2),
            nn.GLU(dim=1)
        )

    def forward(self, x):
        return self.up(x)


class SeStyleMixing(nn.Module):
    def __init__(self, input_channel, output_channel, mode='drop'):
        super(SeStyleMixing, self).__init__()
        assert mode in ['add', 'mu', 'drop']
        self.mode = mode
        self.mapping = nn.Sequential(
            nn.Conv2d(input_channel*2, output_channel*2, kernel_size=(1, 1), ),
            nn.BatchNorm2d(output_channel*2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=output_channel*2, out_channels=output_channel, kernel_size=(1, 1)),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x, se, style):
        # print(x.shape, se.shape, style.shape)
        b, c, h, w = x.shape
        se_feature = x*se
        style_repeat = style.repeat(1, 1, h, w)
        out = self.mapping(torch.cat([se_feature, style_repeat], dim=1))
        # TODO：测试输出要不要加或者乘以原来的x
        if self.mode == 'add':
            return out + x
        elif self.mode == 'mu':
            return out*x
        else:
            return out


class ConditionAttentionLayer(nn.Module):
    def __init__(self, in_channel, hidden_channel=None, mode='add'):
        super(ConditionAttentionLayer, self).__init__()
        assert mode in ['add', 'mu', 'drop']
        self.mode = mode
        if hidden_channel is None:
            hidden_channel = in_channel*2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=(1, 1)),
            nn.ReLU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel*2, hidden_channel, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_channel, out_channels=in_channel, kernel_size=(1, 1))
        )

    def forward(self, x, condition):
        # Bxcx1x1, Bxcxhxw
        b, c, h, w = x.shape
        condition = condition.repeat(1, 1, h, w)
        x1 = self.conv(x)
        # condition = self.conv(condition)
        out = self.conv1(torch.cat([x1, condition], dim=1))
        # TODO: 测试一下这个到底是加号 乘号 还是不要
        if self.mode == 'add':
            return out+x
        elif self.mode == 'mu':
            return out*x
        else:
            return out


class MixingG(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(MixingG, self).__init__()

        self.map1 = nn.Sequential(
            nn.Conv2d(input_channel, out_channels=channels[1], kernel_size=(1, 1), bias=False),
            nn.LeakyReLU(0.1)
        )
        self.map2 = nn.Sequential(
            nn.Conv2d(channels[1], out_channels=channels[2], kernel_size=(1, 1), bias=False),
            nn.LeakyReLU(0.1)
        )
        self.map3 = nn.Sequential(
            nn.Conv2d(channels[2], out_channels=channels[3], kernel_size=(1, 1), bias=False),
            nn.LeakyReLU(0.1)
        )
        self.map4 = nn.Sequential(
            nn.Conv2d(channels[3], out_channels=channels[3], kernel_size=(1, 1), bias=False),
            nn.LeakyReLU(0.1)
        )
        self.map5 = nn.Sequential(
            nn.Conv2d(channels[3], out_channels=channels[4], kernel_size=(1, 1), bias=False),
            nn.LeakyReLU(0.1)
        )

        self.model = nn.Sequential(
            nn.ConvTranspose2d(input_channel, channels[0], 4, 1, 0, bias=True),
            nn.BatchNorm2d(channels[0]),
            nn.ConvTranspose2d(channels[0], channels[0]*2, kernel_size=(1, 1),),
            nn.GLU(dim=1),)

        self.up1 = Upsample(chan_in=channels[0], chan_out=channels[1], scale_factor=2)
        self.ca1 = ConditionAttentionLayer(channels[1])
        self.up2 = Upsample(chan_in=channels[1], chan_out=channels[2])
        self.ca2 = ConditionAttentionLayer(in_channel=channels[2],)

        self.up3 = Upsample(chan_in=channels[2], chan_out=channels[3])
        self.ca3 = ConditionAttentionLayer(channels[3])

        self.up4 = Upsample(chan_in=channels[3], chan_out=channels[3])
        self.semix1 = SeStyleMixing(input_channel=channels[3], output_channel=channels[3])
        self.up5 = Upsample(chan_in=channels[3], chan_out=channels[4])
        self.semix2 = SeStyleMixing(input_channel=channels[4], output_channel=channels[4])

        self.to_rgb = nn.Sequential(
            nn.Conv2d(channels[4], channels[4]*2, kernel_size=(1, 1),),
            nn.BatchNorm2d(channels[4]*2),
            nn.GLU(dim=1),
            nn.Conv2d(channels[4], out_channels=output_channel, kernel_size=(3, 3), padding=(1, 1))
        )

        self.se1 = GlobalContext(chan_in=channels[1], chan_out=channels[3])
        self.se2 = GlobalContext(chan_in=channels[2], chan_out=channels[4])


    def forward(self, x):

        m1 = self.map1(x)
        m2 = self.map2(m1)
        m3 = self.map3(m2)
        m4 = self.map4(m3)
        m5 = self.map5(m4)



        x = self.model(x)
        # print(x.shape)
        x1 = self.up1(x)
        x1 = self.ca1(x1, m1)
        # print(x1.shape)
        se1 = self.se1(x1)
        # print('se1', se1.shape)
        x2 = self.up2(x1)
        x2 = self.ca2(x2, m2)
        # print(x2.shape, 'x2')
        se2 = self.se2(x2)
        # print(se2.shape, 'se2')
        x3 = self.up3(x2)
        x3 = self.ca3(x3, m3)
        # print(x3.shape, 'x3')
        x4 = self.up4(x3)
        x4 = self.semix1(x4, se1, m4)
        # print(x4.shape, 'x4')
        x5 = self.up5(x4)
        x5 = self.semix2(x5, se2, m5)
        # print(x5.shape, 'x5')

        out = self.to_rgb(x5)
        # print(out.shape)
        return out, x2, x3, x5

if __name__ == '__main__':
    a = torch.rand(4, 64, 1, 1)
    g = MixingG(64,  [1024, 512, 256, 128, 64], 3)
    b = g(a)