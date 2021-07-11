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

from parse_config import get_model, get_nips_data_iter

config_name = sys.argv[1]
model_config = importlib.import_module(f'configs.{config_name}')

experiment_name = model_config.exp_name
experiment_path = model_config.exp_path
save_path = os.path.join(experiment_path, experiment_name)
os.makedirs(save_path, exist_ok=True)


print('Parsing your dataset...')
print(model_config.dataset_config['batch_size'])

voice_iter, face_iter = get_nips_data_iter(model_config.dataset_config)

print('Making models')
e_net = get_model(model_config.voice_encoder, False)

g_net = get_model(model_config.generator, False)
g_optimizer = optim.Adam(g_net.parameters(), **model_config.training_config['optimizer'])

d_net = get_model(model_config.discriminator, False)
d_optimizer = optim.Adam(d_net.parameters(), **model_config.training_config['optimizer'])


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
l2_loss = torch.nn.MSELoss().cuda()
affine_loss = torch.nn.KLDivLoss().cuda()
contrastive_loss = SimCLRLoss(temperature=0.2).cuda()
sup_contratsive_loss = SupContrastiveLoss().cuda()
print('Training models...')
for it in range(150000):
    # data
    adjust_learning_rate(optimizer=g_optimizer, epoch=current_epoch, lr=3e-4)
    start_time = time.time()

    voice, voice_label = next(voice_iter)
    face, face_label, face_lr = next(face_iter)
    face = face.cuda()
    voice = voice.cuda()

    embedding = e_net(voice).squeeze()

    data_time.update(time.time() - start_time)

    with torch.no_grad():
        fake, _ = g_net([embedding,])

    for _ in range(3):
        d_optimizer.zero_grad()

        real_score_out = d_net(face)
        fake_score_out = d_net(fake)

        D_loss = dual_contrastive_loss(real_score_out, fake_score_out)

        D_real.update(D_loss.item())
        D_loss.backward()
        d_optimizer.step()

    # Generator
    g_optimizer.zero_grad()
    # arcface_optimizer.zero_grad()

    fake, _ = g_net([embedding, ])
    with torch.no_grad():
        fake_score_out = d_net(fake)
        real_score_out = d_net(face)

    reconstruction_loss = l1_loss(fake, face)
    G_contrastive_loss = dual_contrastive_loss(fake_score_out, real_score_out)

    (G_contrastive_loss + reconstruction_loss).backward()

    torch.nn.utils.clip_grad_norm_(g_net.parameters(), max_norm=2)
    GD_fake.update(G_contrastive_loss.item())
    GC_fake.update(reconstruction_loss.item())
    g_optimizer.step()

    batch_time.update(time.time() - start_time)

    # print status
    if it % 400 == 0:
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

        save_model(g_net, os.path.join(save_path, 'generator.pt'))

    iteration.update(1)

