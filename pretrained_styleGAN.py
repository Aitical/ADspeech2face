import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim

from models import SimCLRLoss, SupContrastiveLoss, dual_contrastive_loss
from utils import Meter, save_model

import math
import sys
import importlib
from models.stylegan2_pytorch import ModelLoader
from configs.criteria import model_paths
from parse_config import get_model, get_edsr, get_vxc_data_iter
from models.voice import StyleMapping
from criteria import LPIPS

config_name = sys.argv[1]
model_config = importlib.import_module(f'configs.{config_name}')
dataset_config = model_config

# NETWORKS_PARAMETERS = config_module.NETWORKS_PARAMETERS
experiment_name = model_config.exp_name
experiment_path = model_config.exp_path
save_path = os.path.join(experiment_path, experiment_name)
os.makedirs(os.path.join(experiment_path, experiment_name), exist_ok=True)
# dataset and dataloader

print('Parsing your dataset...')
data_iter = get_vxc_data_iter(model_config.dataset_config)

print('Making models')
e_net = get_model(model_config.voice_encoder)

g_loader = ModelLoader(
    base_dir=model_paths['stylegan2-pytorch'],   # path to where you invoked the command line tool
    name='default'                   # the project name, defaults to 'default'
)
g_net = g_loader.model.GAN.GE
num_layers = g_loader.model.GAN.GE.num_layers
img_size = g_loader.model.GAN.GE.image_size

style_mapping = StyleMapping(style_dim=512, style_heads=num_layers, mlp_dim=512)
style_mapping.cuda()
s_optimizer = optim.Adam(style_mapping.parameters(), **model_config.training_config['optimizer'])

lpips_loss = LPIPS('vgg')
lpips_loss.eval()
lpips_loss.cuda()

# Meters for recording the training status
iteration = Meter('Iter', 'sum', ':5d')
data_time = Meter('Data', 'sum', ':4.2f')
batch_time = Meter('Time', 'sum', ':4.2f')
D_real = Meter('D_contrastive', 'avg', ':3.2f')
D_fake = Meter('D_fake', 'avg', ':3.2f')
C_real = Meter('C_real', 'avg', ':3.2f')
GD_fake = Meter('G_contrastive', 'avg', ':3.2f')
GC_fake = Meter('G_rec_arc', 'avg', ':3.2f')

current_epoch = 1


def adjust_learning_rate(optimizer, epoch, lr=2e-3):
    """Decay the learning rate based on schedule"""
    # cosine lr schedule
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / 1500))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # wandb.log({'lr': lr, 'epoch': epoch})


l1_loss = torch.nn.L1Loss().cuda()
smooth_l1_loss = torch.nn.SmoothL1Loss().cuda()
l2_loss = torch.nn.MSELoss().cuda()
affine_loss = torch.nn.KLDivLoss().cuda()
contrastive_loss = SimCLRLoss(temperature=0.2).cuda()
sup_contratsive_loss = SupContrastiveLoss().cuda()
print('Training models...')
for it in range(150000):
    # data
    adjust_learning_rate(optimizer=s_optimizer, epoch=current_epoch, lr=3e-3)
    start_time = time.time()

    face, voice, label, face_lr = next(data_iter)

    face = face.cuda()
    voice = voice.cuda()
    label = label.cuda()
    batch_size = voice.shape[0]

    embedding = e_net(voice)
    noise = torch.FloatTensor(batch_size, img_size, img_size, 1).uniform_(0., 1.).cuda()

    data_time.update(time.time() - start_time)

    s_optimizer.zero_grad()
    voice_styles = style_mapping(embedding)
    out_img = g_net(voice_styles, noise)
    prec_loss = lpips_loss(out_img, face)
    rec_loss = smooth_l1_loss(out_img, face)
    # print(rec_loss.item(), prec_loss.item())
    (rec_loss + prec_loss).backward()
    s_optimizer.step()

    batch_time.update(time.time() - start_time)
    D_real.update(rec_loss.item())
    C_real.update(prec_loss.item())
    # print status
    if it % 90 == 0:
        current_epoch += 1
        print(iteration, data_time, batch_time,
              D_real, D_fake, C_real, GD_fake, GC_fake)
        data_time.reset()
        batch_time.reset()
        D_real.reset()
        D_fake.reset()
        C_real.reset()
        GD_fake.reset()
        GC_fake.reset()

        # snapshot

        save_model(style_mapping, os.path.join(save_path, 'stylemapping.pt'))

    iteration.update(1)

