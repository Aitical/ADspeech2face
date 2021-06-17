import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from kornia import filter2d


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


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding = 0, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)


class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)
    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2d(x, f, normalized=True)


class Upsample(nn.Module):
    def __init__(self,chan_in, chan_out, scale_factor=2):
        super(Upsample, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor),
            Blur(),
            nn.Conv2d(chan_in, chan_out * 2, 3, padding=1),
            nn.BatchNorm2d(chan_out * 2),
            nn.GLU(dim=1)
        )

    def forward(self, x):
        return self.up(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, dim_head = 64, heads = 8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.nonlin = nn.GELU()
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, 3, padding = 1, bias = False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, fmap):
        h, x, y = self.heads, *fmap.shape[-2:]
        q, k, v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = h), (q, k, v))

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)

        out = self.nonlin(out)
        return self.to_out(out)


class ConditionAttention(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel=None):
        super(ConditionAttention, self).__init__()
        if hidden_channel is None:
            hidden_channel = in_channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, hidden_channel, kernel_size=(1, 1)),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channel, out_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU()
        )
    def forward(self, x, condition):
        # Bxcx1x1, Bxcxhxw
        x = self.conv1(x)
        p = F.normalize(x, dim=1)
        # Bx1xhxw
        atten_map = F.sigmoid(torch.sum(p*condition, dim=1)).unsqueeze(1)
        out = self.conv2(x*atten_map)
        return out


class ConditionAttenLayer(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel=None):
        super(ConditionAttenLayer, self).__init__()
        if hidden_channel is None:
            hidden_channel = in_channel * 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, hidden_channel, kernel_size=(1, 1)),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channel, out_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU()
        )
    def forward(self, x, condition):
        # Bxcx1x1, Bxcxhxw
        x = self.conv1(x)
        p = F.normalize(x, dim=1)
        # Bx1xhxw
        atten_map = F.sigmoid(torch.sum(p*condition, dim=1)).unsqueeze(1)
        out = self.conv2(x*atten_map)
        return out


class LightG(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(LightG, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(input_channel, channels[0], 4, 1, 0, bias=True),
            nn.BatchNorm2d(channels[0]),
            nn.GLU(dim=1),)

        self.up1 = Upsample(chan_in=channels[0], chan_out=channels[1], scale_factor=2)
        self.up2 = Upsample(chan_in=channels[1], chan_out=channels[1])

        self.up3 = Upsample(chan_in=channels[1], chan_out=channels[2])

        self.up4 = Upsample(chan_in=channels[2], chan_out=channels[3])
        self.up5 = Upsample(chan_in=channels[3], chan_out=channels[4])

        self.to_rgb = nn.Sequential(
            nn.Conv2d(channels[4], channels[4], kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(channels[4]),
            nn.GLU(dim=1),
            nn.Conv2d(channels[4], out_channels=output_channel, kernel_size=(3, 3), padding=(1, 1))
        )

        self.se1 = GlobalContext(chan_in=channels[1], chan_out=channels[3])
        self.se2 = GlobalContext(chan_in=channels[1], chan_out=channels[4])

    def forward(self, x):
        x = self.model(x)
        x1 = self.up1(x)
        se1 = self.se1(x1)

        x2 = self.up2(x1)
        se2 = self.se2(x2)
        x3 = self.up3(x2)
        x4 = self.up4(x3*se1)
        x5 = self.up5(x4*se2)

        out = self.to_rgb(x5)

        return out


class BasicGenerator(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(input_channel, channels[0], 4, 1, 0, bias=True),
            nn.ReLU(),)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(channels[0], channels[1], 4, 2, 1, bias=True),
            nn.ReLU(),)
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(channels[1], channels[2], 4, 2, 1, bias=True),
            nn.ReLU(),)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(channels[2], channels[3], 4, 2, 1, bias=True),
            nn.ReLU(),)
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(channels[3], channels[3], 4, 2, 1, bias=True),
            nn.ReLU()
        )
        self.up6 = nn.Sequential(
            nn.ConvTranspose2d(channels[3], channels[4], 4, 2, 1, bias=True),
            nn.ReLU(),
        )

        self.to_rgb = nn.Sequential(
            nn.Conv2d(channels[4], output_channel, 1, 1, 0, bias=True),
        )
    def forward(self, x):
        # Bx64x1x1
        x = self.model(x)
        x2 = self.up2(x)
        x3 = self.up3(x2)
        x4 = self.up4(x3)
        x5 = self.up5(x4)
        x6 = self.up6(x5)
        x = self.to_rgb(x6)
        return x, x3, x4, x5


class UpSampleModule(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, 4, 2, 1, bias=True),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel*2, kernel_size=(1, 1),),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel, kernel_size=(1, 1)),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2+x1)
        return x3


class BSEGenerator(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=channels[0], kernel_size=(1, 1),),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels[0], out_channels=channels[0]*2, kernel_size=(1, 1), ),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels[0]*2, out_channels=channels[0]*4, kernel_size=(1, 1), ),
            nn.ReLU(),
            nn.ConvTranspose2d(channels[0]*4, channels[0]*4, 4, 1, 0, bias=True),
            nn.ReLU(),)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(channels[0]*4, channels[1], 4, 2, 1, bias=True),
            nn.ReLU(),)
        self.se8to64 = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(in_channels=channels[1], out_channels=channels[1]*2, kernel_size=(4,4), bias=False),
            nn.ReLU(),
            nn.Conv2d(channels[1]*2, channels[3], kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Sigmoid()
        )
        self.se8to128 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=channels[1], out_channels=channels[1]*2,kernel_size=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(channels[1]*2, channels[4], kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Sigmoid()
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(channels[1], channels[2], 4, 2, 1, bias=True),
            nn.ReLU(),)

        self.se16to64 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=channels[2], out_channels=channels[2]*2,kernel_size=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(channels[2]*2, channels[3], kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Sigmoid()
        )
        self.se16to128 = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(in_channels=channels[2], out_channels=channels[2]*2,kernel_size=(4, 4), bias=False),
            nn.ReLU(),
            nn.Conv2d(channels[2]*2, channels[4], kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Sigmoid()
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(channels[2], channels[3], 4, 2, 1, bias=True),
            nn.ReLU(),)
        self.se32to64 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=channels[3], out_channels=channels[3]*2,kernel_size=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(channels[3]*2, channels[3], kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Sigmoid()
        )
        self.se32to128 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=channels[3], out_channels=channels[3]*2,kernel_size=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(channels[3]*2, channels[4], kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Sigmoid()
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(channels[3], channels[3], 4, 2, 1, bias=True),
            nn.ReLU()
        )
        self.up6 = nn.Sequential(
            nn.ConvTranspose2d(channels[3], channels[4], 4, 2, 1, bias=True),
            nn.ReLU(),
        )

        self.to_rgb = nn.Sequential(
            nn.Conv2d(channels[4], output_channel, 1, 1, 0, bias=True),
        )

    def forward(self, x):
        # Bx64x1x1
        x = self.model(x)
        x2 = self.up2(x)
        se_8to64 = self.se8to64(x2)
        # se_8to128 = self.se8to128(x2)

        x3 = self.up3(x2)
        # se_16to64 = self.se16to64(x3)
        se_16to128 = self.se16to128(x3)

        x4 = self.up4(x3)
        # se_32to64 = self.se32to64(x4)
        # se_32to128 = self.se8to128(x4)
        x5 = self.up5(x4)
        x5_se = x5*se_8to64
        x6 = self.up6(x5_se)
        x6_se = x6*se_16to128
        x = self.to_rgb(x6_se)
        return x, x3, x4, x5
