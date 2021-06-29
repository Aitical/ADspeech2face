import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
# from parse_dataset import get_dataset
from models import get_network, SimCLRLoss, SupContrastiveLoss, ResD, dual_contrastive_loss
from utils import Meter, cycle_voice, cycle_face, save_model
from edsr.model import Model
import cv2
from einops import rearrange, repeat
import math
import sys
import importlib

from dataset import VoxCeleb1DataSet, cycle_data
from torchvision.transforms import transforms


config_name = sys.argv[1]
config_module = importlib.import_module(f'config.{config_name}')

dataset_config = config_module.dataset_config
NETWORKS_PARAMETERS = config_module.NETWORKS_PARAMETERS
experiment_name = config_module.experiment_name
experiment_path = config_module.experiment_path

os.makedirs(os.path.join(experiment_path, experiment_name), exist_ok=True)
# dataset and dataloader


print('Parsing your dataset...')


face_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.5, 0.5, 0.5],
                          std=[0.5, 0.5, 0.5])
     ]
)
voice_trans = transforms.Compose(
    [
        torch.tensor
    ]
)
vxc_dataset = VoxCeleb1DataSet(
    root_path=dataset_config['root_path'],
    voice_frame=dataset_config['voice_frame'],
    voice_ext=dataset_config['voice_ext'],
    img_ext=dataset_config['img_ext'],
    voice_transform=voice_trans,
    img_transform=face_transform,
    sample_num=dataset_config['sample_num']
)

dataset_batch_size = dataset_config['batch_size']*dataset_config['sample_num']
NETWORKS_PARAMETERS['c']['output_channel'] = vxc_dataset.num_classes
print(len(vxc_dataset))
train_loader = DataLoader(
    vxc_dataset,
    shuffle=True,
    batch_size=dataset_config['batch_size'],
    num_workers=dataset_config['num_workers'],
    collate_fn=dataset_config['collate_fn']
)

data_iter = cycle_data(train_loader)

# networks, Fe, Fg, Fd (f+d), Fc (f+c)
print('Initializing networks...')
e_net, e_optimizer = get_network('e', NETWORKS_PARAMETERS, train=False)
g_net, g_optimizer = get_network('g', NETWORKS_PARAMETERS, train=True)

d_net = ResD(NETWORKS_PARAMETERS['f']['input_channel'], NETWORKS_PARAMETERS['f']['channels'])
if NETWORKS_PARAMETERS['multi_gpu']:
    d_net = torch.nn.DataParallel(d_net)
d_net.cuda()
d_optimizer = optim.Adam(d_net.parameters(),
                               lr=NETWORKS_PARAMETERS['lr'],
                               betas=(NETWORKS_PARAMETERS['beta1'],NETWORKS_PARAMETERS['beta2']))
sr_model = Model('./pretrained_models/edsr_model/model_best.pt')
sr_model = sr_model.model
sr_model.eval()
for param in sr_model.parameters():
    param.requires_grad = False

if NETWORKS_PARAMETERS['multi_gpu']:
    sr_model = torch.nn.DataParallel(sr_model)
sr_model.cuda()
print('SR model loaded')
# f_net, f_optimizer = get_network('f', NETWORKS_PARAMETERS, train=True)
# d_net, d_optimizer = get_network('d', NETWORKS_PARAMETERS, train=True)
# c_net, c_optimizer = get_network('c', NETWORKS_PARAMETERS, train=True)

arcface, arcface_optimizer = get_network('arcface', NETWORKS_PARAMETERS, train=False)
print('arcface loadded')

# label for real/fake faces
real_label = torch.full((dataset_batch_size, 1), 1)
fake_label = torch.full((dataset_batch_size, 1), 0)

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
    label = label.cuda()
    face_lr = face_lr.cuda()
    # noise = noise.cuda()


    # # use GPU or not
    # if NETWORKS_PARAMETERS['GPU']:
    #     voice, voice_label = voice.cuda(), voice_label.cuda()
    #     print(voice.shape)
    #     face, face_label, face_lr = face.cuda(), face_label.cuda(), face_lr.cuda()
    #     real_label, fake_label = real_label.cuda(), fake_label.cuda()
    #     noise = noise.cuda()

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
    noise = 0.05*torch.rand_like(embeddings, device=embeddings.device)
    # print(embeddings.shape, noise.shape)
    embeddings = embeddings + noise
    embeddings = F.normalize(embeddings)
    real_label = torch.ones((embeddings.shape[0], 1), device=embeddings.device)
    fake_label = torch.zeros_like(real_label, device=embeddings.device)
    # print(embeddings.shape)

    # loss1 = 0.1*(contrastive_loss(embeddings.squeeze(), face_vector) + contrastive_loss(face_vector, embeddings.squeeze()))

    with torch.no_grad():
        fake, _, _, _ = g_net(embeddings)
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

    fake, fake_16, fake_32, fake_64 = g_net(embeddings)
    with torch.no_grad():
        fake_score_out, _ = d_net(fake)
        real_score_out, _ = d_net(face)
    # fake_label_out = c_net(fake)
    # with torch.no_grad():
    # fake_feature_out = F.normalize(f_net(fake).squeeze())
    # real_feature_out = F.normalize(f_net(face).squeeze())
    # print(f_net(fake).shape)

    reconstruction_loss = l1_loss(fake, face)
    arcface_real_embedding = arcface(face)
    arcface_fake_embedding = arcface(fake)
    arcface_loss = l2_loss(F.normalize(arcface_fake_embedding, dim=1), F.normalize(arcface_real_embedding, dim=1))
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

        save_model(g_net.module, NETWORKS_PARAMETERS['g']['model_path'], NETWORKS_PARAMETERS['multi_gpu'])


        # save_model(e_net, NETWORKS_PARAMETERS['e']['model_path'])
        # save_model(f_net, NETWORKS_PARAMETERS['f']['model_path'])
        # save_model(d_net, NETWORKS_PARAMETERS['d']['model_path'])
        # save_model(c_net, NETWORKS_PARAMETERS['c']['model_path'])
    iteration.update(1)

