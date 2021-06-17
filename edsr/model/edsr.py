from . import common

import torch.nn as nn

url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}

def make_model(args, parent=False):
    return EDSR(args)

class EDSR(nn.Module):
    def __init__(self, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = 16
        n_feats = 64
        kernel_size = 3
        scale = 4
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = common.MeanShift(255)
        self.add_mean = common.MeanShift(255, sign=1)

        # define head module
        m_head = [conv(3, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=1
            ) for _ in range(n_resblocks)
        ]
        self.my_body_head = conv(n_feats, 256, kernel_size)

        # define tail module
        # m_tail = [
        #     common.Upsampler(conv, scale, n_feats, act=False),
        #     conv(n_feats, args.n_colors, kernel_size)
        # ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        self.up1 = nn.Sequential(
            conv(256, 512, 3, bias=True),
            nn.PReLU(512),
            conv(512, 512, 3, bias=True),
            nn.PixelShuffle(2),
            nn.PReLU(128)
        )

        self.up2 = nn.Sequential(
            conv(128, 512, 3, bias=True),
            nn.PReLU(512),
            conv(512, 512, 3, bias=True),
            nn.PixelShuffle(2),
            nn.PReLU(128)
        )
        self.up3 = nn.Sequential(
            conv(128, 256, 3, bias=True),
            nn.PReLU(256),
            conv(256, 256, 3, bias=True),
            nn.PixelShuffle(2),
            nn.PReLU(64)
        )
        self.to_rgb = conv(64, 3, kernel_size)
        # self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x16 = self.my_body_head(res)
        x1 = self.up1(x16)
        x2 = self.up2(x1)
        x3 = self.up3(x2)
        x = self.to_rgb(x3)
        x = self.add_mean(x)
        return res, x16, x1, x2

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

