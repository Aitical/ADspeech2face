import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from kornia.filters.filter import filter2D as filter2d
from .voice import ResNetSE34

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


class LinearConditionAttention(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel=None):
        super(LinearConditionAttention, self).__init__()
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


class ConditionAttentionLayer(nn.Module):
    def __init__(self, in_channel, hidden_channel=None):
        super(ConditionAttentionLayer, self).__init__()
        if hidden_channel is None:
            hidden_channel = in_channel*2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=256, kernel_size=(1, 1)),
            nn.ReLU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(512, hidden_channel, kernel_size=(1, 1)),
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
        return out+x


class VLightG(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(VLightG, self).__init__()
        self.voice = ResNetSE34()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(input_channel, channels[0], 4, 1, 0, bias=True),
            nn.BatchNorm2d(channels[0]),
            nn.ConvTranspose2d(channels[0], channels[0]*2, kernel_size=(1, 1),),
            nn.GLU(dim=1),)

        self.up1 = Upsample(chan_in=channels[0], chan_out=channels[1], scale_factor=2)
        self.up2 = Upsample(chan_in=channels[1], chan_out=channels[2])

        self.up3 = Upsample(chan_in=channels[2], chan_out=channels[3])

        self.up4 = Upsample(chan_in=channels[3], chan_out=channels[3])
        self.up5 = Upsample(chan_in=channels[3], chan_out=channels[4])

        self.to_rgb = nn.Sequential(
            nn.Conv2d(channels[4], channels[4]*2, kernel_size=(1, 1),),
            nn.BatchNorm2d(channels[4]*2),
            nn.GLU(dim=1),
            nn.Conv2d(channels[4], out_channels=output_channel, kernel_size=(3, 3), padding=(1, 1))
        )

        self.se1 = GlobalContext(chan_in=channels[1], chan_out=channels[3])
        self.se2 = GlobalContext(chan_in=channels[2], chan_out=channels[4])

    def forward(self, x):
        x = self.voice(x)
        x = self.model(x)
        # print(x.shape)
        x1 = self.up1(x)
        # print(x1.shape)
        se1 = self.se1(x1)
        # print('se1', se1.shape)
        x2 = self.up2(x1)
        # print(x2.shape, 'x2')
        se2 = self.se2(x2)
        # print(se2.shape, 'se2')
        x3 = self.up3(x2)
        # print(x3.shape, 'x3')
        x4 = self.up4(x3)
        # print(x4.shape, 'x4')
        x5 = self.up5(x4*se1)
        # print(x5.shape, 'x5')

        out = self.to_rgb(x5*se2)
        # print(out.shape)
        return out, x2, x3, x4

class LightG(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(LightG, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(input_channel, channels[0], 4, 1, 0, bias=True),
            nn.BatchNorm2d(channels[0]),
            nn.ConvTranspose2d(channels[0], channels[0]*2, kernel_size=(1, 1),),
            nn.GLU(dim=1),)

        self.up1 = Upsample(chan_in=channels[0], chan_out=channels[1], scale_factor=2)
        self.up2 = Upsample(chan_in=channels[1], chan_out=channels[2])

        self.up3 = Upsample(chan_in=channels[2], chan_out=channels[3])

        self.up4 = Upsample(chan_in=channels[3], chan_out=channels[3])
        self.up5 = Upsample(chan_in=channels[3], chan_out=channels[4])

        self.to_rgb = nn.Sequential(
            nn.Conv2d(channels[4], channels[4]*2, kernel_size=(1, 1),),
            nn.BatchNorm2d(channels[4]*2),
            nn.GLU(dim=1),
            nn.Conv2d(channels[4], out_channels=output_channel, kernel_size=(3, 3), padding=(1, 1))
        )

        self.se1 = GlobalContext(chan_in=channels[1], chan_out=channels[3])
        self.se2 = GlobalContext(chan_in=channels[2], chan_out=channels[4])

    def forward(self, x):
        x = self.model(x)
        # print(x.shape)
        x1 = self.up1(x)
        # print(x1.shape)
        se1 = self.se1(x1)
        # print('se1', se1.shape)
        x2 = self.up2(x1)
        # print(x2.shape, 'x2')
        se2 = self.se2(x2)
        # print(se2.shape, 'se2')
        x3 = self.up3(x2)
        # print(x3.shape, 'x3')
        x4 = self.up4(x3)
        # print(x4.shape, 'x4')
        x5 = self.up5(x4*se1)
        # print(x5.shape, 'x5')

        out = self.to_rgb(x5*se2)
        # print(out.shape)
        return out, x2, x3, x4
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


class UpConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.1)
        )
    def forward(self, x):
        out = self.up(x)
        return out


class ResUpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResUpConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
        )
        self.conv2 = UpConv(in_channel=in_channels, out_channel=out_channels)
        self.conv = DepthWiseConv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        # print(x1.shape, x2.shape)
        out = self.conv(x1+x2)
        return out


class ResG(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=128, bias=False, kernel_size=(1, 1)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=128, out_channels=256, bias=False, kernel_size=(1, 1)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=256, bias=False, kernel_size=(1, 1)),
            nn.LeakyReLU(0.1),
            )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, channels[0], 4, 1, 0, bias=True),
            nn.ReLU(),
        )
        self.up2 = ResUpConv(in_channels=channels[0], out_channels=channels[1])
        self.up3 = ResUpConv(in_channels=channels[1], out_channels=channels[2])
        self.ca1 = ConditionAttentionLayer(in_channel=channels[2])
        self.up4 = ResUpConv(in_channels=channels[2], out_channels=channels[3])
        self.up5 = ResUpConv(in_channels=channels[3], out_channels=channels[3])
        self.ca2 = ConditionAttentionLayer(in_channel=channels[3])
        self.up6 = ResUpConv(in_channels=channels[3], out_channels=channels[4])

        self.to_rgb = nn.Sequential(
            nn.Conv2d(channels[4], output_channel, 1, 1, 0, bias=True),
        )

    def forward(self, x):
        # Bx64x1x1
        c = self.model(x)
        x1 = self.up1(c)
        x2 = self.up2(x1)
        x3 = self.up3(x2)
        # print(x3.shape, c.shape)
        x3 = self.ca1(x3, c)
        x4 = self.up4(x3)
        x5 = self.up5(x4)
        x5 = self.ca2(x5, c)
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

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(channels[1], channels[2], 4, 2, 1, bias=True),
            nn.ReLU(),)

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


class ResDownConv(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(ResDownConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, input_channel, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(input_channel, output_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(0.1),
        )
        self.avg = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Conv2d(input_channel, output_channel, kernel_size=(1, 1)),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.avg(x)
        return x1+x2


class SimpleDecoder(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(SimpleDecoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(input_channel, output_channel*2, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(output_channel*2),
            nn.GLU(dim=1)
        )
        self.conv2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(output_channel, output_channel*2, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(output_channel*2),
            nn.GLU(dim=1)
        )
        self.conv3 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(output_channel, output_channel*2, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(output_channel*2),
            nn.GLU(dim=1)
        )

        self.to_rgb = nn.Sequential(
            nn.Conv2d(output_channel, 3, kernel_size=(1, 1))
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        out = self.to_rgb(x3)
        return out


class ResD(nn.Module):
    def __init__(self, input_channel, channels,):
        super().__init__()

        self.res64 = ResDownConv(input_channel, channels[0])
        self.res32 = ResDownConv(channels[0], channels[1])
        self.res16 = ResDownConv(channels[1], channels[2])
        self.res8 = ResDownConv(channels[2], channels[3])

        self.to_logits = nn.Sequential(
            nn.Conv2d(channels[3], channels[4], kernel_size=(4, 4)),
            nn.BatchNorm2d(channels[4]),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels[4], 1, kernel_size=(1, 1)),
            nn.Flatten()
        )
        self.decoder = SimpleDecoder(channels[2], channels[2])

    def forward(self, x, ):
        x64 = self.res64(x)
        x32 = self.res32(x64)
        x16 = self.res16(x32)

        x8 = self.res8(x16)
        logits = self.to_logits(x8)

        out = self.decoder(x16)

        return logits, out


def dual_contrastive_loss(real_logits, fake_logits):
    device = real_logits.device
    real_logits, fake_logits = map(lambda t: rearrange(t, '... -> (...)'), (real_logits, fake_logits))

    def loss_half(t1, t2):
        t1 = rearrange(t1, 'i -> i ()')
        t2 = repeat(t2, 'j -> i j', i = t1.shape[0])
        t = torch.cat((t1, t2), dim = -1)
        return F.cross_entropy(t, torch.zeros(t1.shape[0], device = device, dtype = torch.long))

    return loss_half(real_logits, fake_logits) + loss_half(-fake_logits, -real_logits)

def gen_hinge_loss(fake, real):
    return fake.mean()

def hinge_loss(real, fake):
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()

if __name__ == '__main__':
    a = torch.rand(3, 64, 1, 1)
    m = ResG(64, [1024, 512, 256, 128, 64], 3)
    b = m(a)
    print(b[0].shape)