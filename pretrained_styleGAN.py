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
from stylegan2_pytorch import ModelLoader
from parse_config import get_model, get_edsr, get_nips_data_iter

config_name = sys.argv[1]
model_config = importlib.import_module(f'configs.{config_name}')
dataset_config = model_config

# NETWORKS_PARAMETERS = config_module.NETWORKS_PARAMETERS
experiment_name = model_config.exp_name
experiment_path = model_config.exp_path
save_path = os.path.join(experiment_path, experiment_name)
os.makedirs(os.path.join(experiment_path, experiment_name), exist_ok=True)
# dataset and dataloader

voice_iter, face_iter = get_nips_data_iter(dataset_config)

print('Making models')
e_net = get_model(model_config.voice_encoder)


g_net = get_model(model_config.generator)
g_optimizer = optim.Adam(g_net.parameters(), **model_config.training_config['optimizer'])

d_net = get_model(model_config.discriminator)
d_optimizer = optim.Adam(d_net.parameters(), **model_config.training_config['optimizer'])

arcface = get_model(model_config.arcface)
sr_model = get_edsr()
print('Model Prepared')


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
l2_loss = torch.nn.MSELoss().cuda()
affine_loss = torch.nn.KLDivLoss().cuda()
contrastive_loss = SimCLRLoss(temperature=0.2).cuda()
sup_contratsive_loss = SupContrastiveLoss().cuda()
print('Training models...')
for it in range(150000):
    # data
    adjust_learning_rate(optimizer=g_optimizer, epoch=current_epoch, lr=3e-4)
    start_time = time.time()

    face, voice, label, face_lr = next(data_iter)

    face = face.cuda()
    voice = voice.cuda()
    # print(voice.shape, 'voice')
    label = label.cuda()
    face_lr = face_lr.cuda()
    # noise = noise.cuda()
    voice = e_net(voice)

    data_time.update(time.time() - start_time)

    with torch.no_grad():
        latent, lr_16, lr_32, lr_64 = sr_model(face_lr)
        # print(latent.shape, lr_16.shape, lr_64.shape)
        # BXCXHxH
        face_vector = torch.mean(lr_16, dim=[2, 3])
        face_vector = torch.nn.functional.normalize(face_vector, dim=1)

    real_label = torch.ones((voice.shape[0], 1), device=voice.device)
    fake_label = torch.zeros_like(real_label, device=voice.device)
    # print(embeddings.shape)

    # loss1 = 0.1*(contrastive_loss(embeddings.squeeze(), face_vector) + contrastive_loss(face_vector, embeddings.squeeze()))

    with torch.no_grad():
        fake, _, _, _ = g_net(voice)
    # print(fake.shape, fake_16.shape, fake_64.shape)
    # print(fake.shape)
    # Discriminator
    # e_optimizer.zero_grad()
    # f_optimizer.zero_grad()
    d_optimizer.zero_grad()
    # c_optimizer.zero_grad()
    # arcface_optimizer.zero_grad()
    # arcface_optimizer.zero_grad()

    real_score_out, real_rec = d_net(face)
    fake_score_out, fake_rec = d_net(fake)

    real_rec_embd = arcface(real_rec)
    fake_rec_embd = arcface(fake_rec)

    # real_label_out = c_net(f_net(face))
    # clip_feature = F.normalize(f_net(face).squeeze())
    #  print(clip_feature.shape, embeddings.shape)
    #
    # F_clip_loss = 0.1 * 0.5*(contrastive_loss(clip_feature, embeddings.squeeze().detach()) + contrastive_loss(embeddings.squeeze().detach(), clip_feature))
    # clip_fake_feature = F.normalize(f_net(fake.detach()).squeeze())
    # F_clip_contrastive = 0.3 * contrastive_loss(clip_fake_feature, clip_feature)

    # D_real_loss = F.binary_cross_entropy(torch.sigmoid(real_score_out), real_label.float())
    # D_fake_loss = F.binary_cross_entropy(torch.sigmoid(fake_score_out), fake_label.float())
    # C_real_loss = F.nll_loss(F.log_softmax(real_label_out, 1), label)
    D_loss = dual_contrastive_loss(real_score_out, fake_score_out)
    D_arcface_loss = l2_loss(F.normalize(fake_rec_embd, dim=1), F.normalize(real_rec_embd, dim=1))
    D_rec_loss = l1_loss(real_rec, face)
    # reconstruction_loss = l1_loss()

    D_real.update(D_loss.item())
    C_real.update(D_arcface_loss.item())
    (D_loss + 0.3*D_rec_loss + 0.3*D_arcface_loss).backward()
    # f_optimizer.step()
    d_optimizer.step()
    # c_optimizer.step()

    # Generator
    g_optimizer.zero_grad()
    # arcface_optimizer.zero_grad()

    fake, fake_16, fake_32, fake_64 = g_net(voice)
    with torch.no_grad():
        fake_score_out, _ = d_net(fake)
        real_score_out, _ = d_net(face)
    # fake_label_out = c_net(fake)
    # with torch.no_grad():
    # fake_feature_out = F.normalize(f_net(fake).squeeze())
    # real_feature_out = F.normalize(f_net(face).squeeze())
    # print(f_net(fake).shape)

    reconstruction_loss = l1_loss(fake, face)
    arcface_real_embedding = F.normalize(arcface(face), dim=1)
    arcface_fake_embedding = F.normalize(arcface(fake), dim=1)
    # arcface_loss = l2_loss(F.normalize(arcface_fake_embedding, dim=1), F.normalize(arcface_real_embedding, dim=1))
    arcface_loss = l2_loss(arcface_fake_embedding, arcface_real_embedding)
    G_contrastive_loss = dual_contrastive_loss(fake_score_out, real_score_out)

    # GD_fake_loss = F.binary_cross_entropy(torch.sigmoid(fake_score_out), real_label.float())
    # GC_fake_loss = 0.5 * F.nll_loss(F.log_softmax(fake_label_out, 1), label)
    # Embedded_contrastive_loss = 0.5 * sup_contratsive_loss(fake_feature_out, real_feature_out, voice_label)
    # Embedded_contrastive_loss = 0.1 * l2_loss(fake_feature_out, real_feature_out)
    # out_space_loss = 0.1*(0.5*l1_loss(fake_16, lr_16) + 0.5*l1_loss(fake_32, lr_32))

    # loss2 = 0.1 * (l1_loss(fake_16, lr_16))
    loss32 = 0.1 * (l1_loss(fake_32, lr_32)) + 0.1*(l1_loss(fake_16, lr_16))
    # # BxCx16x16
    # b, c, h, w = lr_16.shape
    # non_local_lr = lr_16.reshape(b, c, h*w)
    # non_local_sim = torch.bmm(non_local_lr.permute(0, 2, 1), non_local_lr).reshape(b*h*w, h*w)
    # non_local_prob = torch.nn.functional.softmax(non_local_sim, dim=1)
    #
    #
    # non_local_fake = fake_16.reshape(b, c, h*w)
    # non_local_sim_fake = torch.bmm(non_local_fake.permute(0, 2, 1), non_local_fake).reshape(b*h*w, h*w)
    # non_local_fake_prob = torch.nn.functional.log_softmax(non_local_sim_fake, dim=1)
    #
    # loss2 = 0.05 * affine_loss(non_local_fake_prob, non_local_prob)

    (0.2 * G_contrastive_loss + 0.3*reconstruction_loss + 0.2 * arcface_loss + 0.2*loss32).backward()
    torch.nn.utils.clip_grad_norm_(g_net.parameters(), max_norm=1)
    GD_fake.update(G_contrastive_loss.item())
    GC_fake.update(reconstruction_loss.item() + arcface_loss.item())
    g_optimizer.step()
    # e_optimizer.step()
    batch_time.update(time.time() - start_time)

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

        save_model(g_net.module, save_path)

    iteration.update(1)

