import os
import time
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from config import DATASET_PARAMETERS, NETWORKS_PARAMETERS
from parse_dataset import get_dataset
from network import get_network, SimCLRLoss, SupContrastiveLoss
from utils import Meter, cycle_voice, cycle_face, save_model
from edsr.model import Model
import cv2
from einops import rearrange, repeat
import math


# dataset and dataloader
print('Parsing your dataset...')
voice_list, face_list, id_class_num = get_dataset(DATASET_PARAMETERS)
NETWORKS_PARAMETERS['c']['output_channel'] = id_class_num

print('Preparing the datasets...')
voice_dataset = DATASET_PARAMETERS['voice_dataset'](voice_list,
                               DATASET_PARAMETERS['nframe_range'])
face_dataset = DATASET_PARAMETERS['face_dataset'](face_list)

print('Preparing the dataloaders...')
collate_fn = DATASET_PARAMETERS['collate_fn'](DATASET_PARAMETERS['nframe_range'])
voice_loader = DataLoader(voice_dataset, shuffle=True, drop_last=True,
                          batch_size=DATASET_PARAMETERS['batch_size'],
                          num_workers=DATASET_PARAMETERS['workers_num'],
                          collate_fn=collate_fn)
face_loader = DataLoader(face_dataset, shuffle=True, drop_last=True,
                         batch_size=DATASET_PARAMETERS['batch_size'],
                         num_workers=DATASET_PARAMETERS['workers_num'])

voice_iterator = iter(cycle_voice(voice_loader))
face_iterator = iter(cycle_face(face_loader))

# networks, Fe, Fg, Fd (f+d), Fc (f+c)
print('Initializing networks...')
e_net, e_optimizer = get_network('e', NETWORKS_PARAMETERS, train=False)
g_net, g_optimizer = get_network('g', NETWORKS_PARAMETERS, train=True)
f_net, f_optimizer = get_network('f', NETWORKS_PARAMETERS, train=True)
d_net, d_optimizer = get_network('d', NETWORKS_PARAMETERS, train=True)
c_net, c_optimizer = get_network('c', NETWORKS_PARAMETERS, train=True)

# label for real/fake faces
real_label = torch.full((DATASET_PARAMETERS['batch_size'], 1), 1)
fake_label = torch.full((DATASET_PARAMETERS['batch_size'], 1), 0)

# Meters for recording the training status
iteration = Meter('Iter', 'sum', ':5d')
data_time = Meter('Data', 'sum', ':4.2f')
batch_time = Meter('Time', 'sum', ':4.2f')
D_real = Meter('D_real', 'avg', ':3.2f')
D_fake = Meter('D_fake', 'avg', ':3.2f')
C_real = Meter('C_real', 'avg', ':3.2f')
GD_fake = Meter('G_D_fake', 'avg', ':3.2f')
GC_fake = Meter('G_C_fake', 'avg', ':3.2f')

current_epoch = 1

def adjust_learning_rate(optimizer, epoch, lr=0.1):
    """Decay the learning rate based on schedule"""
      # cosine lr schedule
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / 400))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # wandb.log({'lr': lr, 'epoch': epoch})

sr_model = Model('/home/aitical/Documents/paper_with_code/speech2face/speech2face_dense/edsr/pretrained/model_best.pt')
sr_model.model.eval()
for param in sr_model.model.parameters():
    param.requires_grad = False

sr_model.cuda()
print('SR model loaded')
l1_loss = torch.nn.L1Loss().cuda()
l2_loss = torch.nn.MSELoss().cuda()
affine_loss = torch.nn.KLDivLoss().cuda()
contrastive_loss = SimCLRLoss(temperature=0.2).cuda()
sup_contratsive_loss = SupContrastiveLoss().cuda()
print('Training models...')
for it in range(50000):
    # data
    adjust_learning_rate(optimizer=g_optimizer, epoch=current_epoch, lr=3e-4)
    start_time = time.time()
    
    voice, voice_label = next(voice_iterator)
    face, face_label, face_lr = next(face_iterator)
    noise = 0.05*torch.randn(DATASET_PARAMETERS['batch_size'], NETWORKS_PARAMETERS['e']['output_channel'], 1, 1)

    # use GPU or not
    if NETWORKS_PARAMETERS['GPU']: 
        voice, voice_label = voice.cuda(), voice_label.cuda()
        face, face_label, face_lr = face.cuda(), face_label.cuda(), face_lr.cuda()
        real_label, fake_label = real_label.cuda(), fake_label.cuda()
        noise = noise.cuda()

    data_time.update(time.time() - start_time)

    with torch.no_grad():
        latent, lr_16, lr_32, lr_64 = sr_model(face_lr)
        # print(latent.shape, lr_16.shape, lr_64.shape)
        # BXCXHxH
        face_vector = torch.mean(lr_16, dim=[2, 3])
        face_vector = torch.nn.functional.normalize(face_vector, dim=1)

    # get embeddings and generated faces
    embeddings = e_net(voice)
    embeddings = F.normalize(embeddings)
    # introduce some permutations
    embeddings = embeddings + noise
    embeddings = F.normalize(embeddings)
    # print(embeddings.shape)

    # loss1 = 0.1*(contrastive_loss(embeddings.squeeze(), face_vector) + contrastive_loss(face_vector, embeddings.squeeze()))

    fake, fake_16, fake_32, fake_64 = g_net(embeddings)
    # print(fake.shape, fake_16.shape, fake_64.shape)
    # print(fake.shape)
    # Discriminator
    # e_optimizer.zero_grad()
    f_optimizer.zero_grad()
    d_optimizer.zero_grad()
    c_optimizer.zero_grad()
    real_score_out = d_net(f_net(face))
    fake_score_out = d_net(f_net(fake.detach()))
    real_label_out = c_net(f_net(face))
    clip_feature = F.normalize(f_net(face).squeeze())
   #  print(clip_feature.shape, embeddings.shape)
    #
    # F_clip_loss = 0.1 * 0.5*(contrastive_loss(clip_feature, embeddings.squeeze().detach()) + contrastive_loss(embeddings.squeeze().detach(), clip_feature))
    # clip_fake_feature = F.normalize(f_net(fake.detach()).squeeze())
    # F_clip_contrastive = 0.3 * contrastive_loss(clip_fake_feature, clip_feature)

    D_real_loss = F.binary_cross_entropy(torch.sigmoid(real_score_out), real_label.float())
    D_fake_loss = F.binary_cross_entropy(torch.sigmoid(fake_score_out), fake_label.float())
    C_real_loss = F.nll_loss(F.log_softmax(real_label_out, 1), face_label)
    D_real.update(D_real_loss.item())
    D_fake.update(D_fake_loss.item())
    C_real.update(C_real_loss.item())
    (D_real_loss + D_fake_loss + C_real_loss).backward()
    f_optimizer.step()
    d_optimizer.step()
    c_optimizer.step()

    # Generator
    g_optimizer.zero_grad()
    fake_score_out = d_net(f_net(fake))
    fake_label_out = c_net(f_net(fake))
    # with torch.no_grad():
    fake_feature_out = F.normalize(f_net(fake).squeeze())
    real_feature_out = F.normalize(f_net(face).squeeze())
    # print(f_net(fake).shape)

    GD_fake_loss = F.binary_cross_entropy(torch.sigmoid(fake_score_out), real_label.float())
    GC_fake_loss = 0.5 * F.nll_loss(F.log_softmax(fake_label_out, 1), voice_label)
    # Embedded_contrastive_loss = 0.5 * sup_contratsive_loss(fake_feature_out, real_feature_out, voice_label)
    # Embedded_contrastive_loss = 0.1 * l2_loss(fake_feature_out, real_feature_out)
    # out_space_loss = 0.1*(0.5*l1_loss(fake_16, lr_16) + 0.5*l1_loss(fake_32, lr_32))

    loss2 = 0.1*(l1_loss(fake_16, lr_16))
    loss3 = 0.1*(l1_loss(fake_64, lr_64))
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

    (GD_fake_loss + GC_fake_loss + loss2 + loss3).backward()
    GD_fake.update(GD_fake_loss.item())
    GC_fake.update(GC_fake_loss.item())
    g_optimizer.step()
    # e_optimizer.step()
    batch_time.update(time.time() - start_time)

    # print status
    if it % 200 == 0:
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
        save_model(g_net, NETWORKS_PARAMETERS['g']['model_path'])
        # save_model(e_net, NETWORKS_PARAMETERS['e']['model_path'])
        save_model(f_net, NETWORKS_PARAMETERS['f']['model_path'])
        save_model(d_net, NETWORKS_PARAMETERS['d']['model_path'])
        save_model(c_net, NETWORKS_PARAMETERS['c']['model_path'])
    iteration.update(1)

