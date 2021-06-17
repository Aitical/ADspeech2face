import torch
import torch.nn as nn


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
