import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class VoiceEmbedNet(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(VoiceEmbedNet, self).__init__()
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
        )

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool1d(x, x.size()[2], stride=1)
        x = x.view(x.size()[0], -1, 1, 1)
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride, 1, bias=True),
            nn.ReLU()
        )
        self.dense1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.dense2 = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels*3, out_channels, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        d1 = self.dense1(x1)
        d2 = self.dense2(torch.cat([d1, x1], dim=1))
        x3 = self.conv2(torch.cat([d2, d1, x1], dim=1))
        return x3+x1

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



class LayerAttenGenerator(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(LayerAttenGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(input_channel, channels[0], 4, 1, 0, bias=True),
            nn.ReLU(inplace=True),)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(channels[0], channels[1], 4, 2, 1, bias=True),
            nn.ReLU(inplace=True),)
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(channels[1], channels[2], 4, 2, 1, bias=True),
            nn.ReLU(inplace=True),)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(channels[2], channels[3], 4, 2, 1, bias=True),
            nn.ReLU(inplace=True),)
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(channels[3], channels[4], 4, 2, 1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.to_rgb = nn.Sequential(
            nn.ConvTranspose2d(channels[4], output_channel, 1, 1, 0, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        x2 = self.up2(x)
        x3 = self.up3(x2)
        x4 = self.up4(x3)
        x5 = self.up5(x4)
        x = self.to_rgb(x5)
        return x, x3, x4, x5

class AttnGenerator(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(AttnGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(input_channel, channels[0], 4, 1, 0, bias=True),
            nn.ReLU(inplace=True),)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(channels[0], channels[1], 4, 2, 1, bias=True),
            nn.ReLU(inplace=True),)
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(channels[1], channels[2], 4, 2, 1, bias=True),
            nn.ReLU(inplace=True),)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(channels[2], channels[3], 4, 2, 1, bias=True),
            nn.ReLU(inplace=True),)
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(channels[3], channels[3], 4, 2, 1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.up6 = nn.Sequential(
            nn.ConvTranspose2d(channels[3], channels[4], 4, 2, 1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(channels[4], channels[4], kernel_size=(1, 1)),
            nn.ReLU()
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
        p6 = self.projection(x6)
        x_norm = F.normalize(x, dim=1)
        p6_norm = F.normalize(p6, dim=1)
        # Bx64x128x128
        atten_map = torch.sum(x_norm * p6_norm, dim=1).unsqueeze(1)
        a6 = x6 * atten_map + x6
        x = self.to_rgb(a6)
        return x, x3, x4, x5


class DenseGenerator(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(DenseGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(input_channel, channels[0], 4, 1, 0, bias=True),
            nn.ReLU(inplace=True),)
        self.up2 = DenseBlock(channels[0], channels[1])
        self.up3 = DenseBlock(channels[1], channels[2])
        self.up4 = DenseBlock(channels[2], channels[3])
        self.up5 = DenseBlock(channels[3], channels[3])
        self.up6 = DenseBlock(channels[3], channels[4])
        self.to_rgb = nn.Sequential(
            nn.Conv2d(channels[4], output_channel, kernel_size=(1, 1))
        )
    def forward(self, x):
        x = self.model(x)
        x2 = self.up2(x)
        x3 = self.up3(x2)
        x4 = self.up4(x3)
        x5 = self.up5(x4)
        x6 = self.up6(x5)
        x = self.to_rgb(x6)
        return x, x3, x4, x5


class Generator(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(Generator, self).__init__()
        self.model1 = nn.Sequential(
                        nn.ConvTranspose2d(input_channel, channels[0], 4, 1, 0, bias=True),
                        nn.ReLU(inplace=True),)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(input_channel, channels[0], 4, 1, 0, bias=True),
            nn.ReLU(inplace=True),)
        self.up2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[1], channels[1]*4, kernel_size=(3, 3), padding=(1, 1), bias=True),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),)
        self.up3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[2], channels[2]*4, kernel_size=(3, 3), padding=(1, 1), bias=True),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),)
        self.up4 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], kernel_size=(3, 3), padding=(1, 1),),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[3], channels[3]*4, kernel_size=(3, 3), padding=(1, 1), bias=True),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),)
        self.up5 = nn.Sequential(
            nn.Conv2d(channels[3], channels[3], kernel_size=(3, 3), padding=(1, 1),),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[3], channels[3]*4, kernel_size=(3, 3), padding=(1, 1), bias=True),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        self.to_rgb = nn.Sequential(
            nn.Conv2d(channels[3], channels[4], kernel_size=(3, 3), padding=(1, 1),),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[4], channels[4] * 4, kernel_size=(1, 1), ),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[4], output_channel, kernel_size=(3, 3), padding=(1, 1), bias=True),
        )

    def forward(self, x):
        # print(self.model1(x).shape)
        x = self.model(x)
        # print(x.shape, 'x')
        x2 = self.up2(x)
        # print(x2.shape, 'x2')
        x3 = self.up3(x2)
        # print(x3.shape, 'x3')
        x4 = self.up4(x3)
        # print(x4.shape, 'x4')
        x5 = self.up5(x4)
        # print(x5.shape, 'x5')
        x = self.to_rgb(x5)
        # print(x.shape, 'x6')
        return x, x3, x4, x5


class FaceEmbedNet(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(FaceEmbedNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channel, channels[0], 1, 1, 0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels[0], channels[0], 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels[0], channels[1], 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels[1], channels[2], 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels[2], channels[3], 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels[3], channels[4], 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels[4], output_channel, 4, 1, 0, bias=True),

        )
 
    def forward(self, x):
        x = self.model(x)
        return x

class Classifier(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(Classifier, self).__init__()
        self.model = nn.Linear(input_channel, output_channel, bias=False)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.model(x)
        return x

def get_network(net_type, params, train=True):
    net_params = params[net_type]
    net = net_params['network'](net_params['input_channel'],
                                net_params['channels'],
                                net_params['output_channel'])

    if params['GPU']:
        net.cuda()

    if train:
        net.train()
        optimizer = optim.Adam(net.parameters(),
                               lr=params['lr'],
                               betas=(params['beta1'], params['beta2']))
    else:
        net.eval()
        net.load_state_dict(torch.load(net_params['model_path']))
        optimizer = None
    return net, optimizer


class SimCLRLoss(nn.Module):
    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()

        self.T = temperature
        self.ce = nn.CrossEntropyLoss()
        self.norm = nn.functional.normalize
        self.softmax = nn.functional.softmax
        self.cosine = nn.CosineSimilarity(dim=-1)
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.mse = nn.MSELoss()

    def _get_correlated_mask(self, batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = diag + l1 + l2
        # mask = (1 - mask).type(torch.bool)
        return mask

    def forward(self, f1, f2):
        batch_size = f1.shape[0]
        # f = torch.cat([f1, f2], dim=0)

        sim_matrix = self.cosine(f1.unsqueeze(1), f2.unsqueeze(0)) / self.T
        label = torch.arange(0, batch_size, device=sim_matrix.device)

        loss = self.ce(sim_matrix, label) + self.ce(sim_matrix.t(), label)
        return loss * 0.5


class SupContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.t = temperature

    def forward(self, x1, x2, label):
        # x1 = x1.view(x1.size()[0], -1)
        # x2 = x2.view(x2.size()[0], -1)
        # BxB
        sim = torch.exp(x1.mm(x2.t()) / self.t)
        # print(sim.shape)
        label = label.reshape(-1, 1)
        # BxB
        label_matrix = label.eq(label.t()).float()
        reg_value = 1 / label_matrix.sum(dim=1, keepdim=True)
        logits = -torch.log(sim/sim.sum(dim=1, keepdim=True)) * label_matrix
        loss = torch.sum(logits)
        return loss

if __name__ == '__main__':
    model = DenseGenerator(64,[128, 128, 128, 128, 128], 3)
    a = torch.rand(16, 64, 1, 1)
    f = model(a)