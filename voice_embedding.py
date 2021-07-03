import random

from dataset import VoiceDataSet
from configs.voice import dataset_config
import torch
from models import resnet50, resnet18
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from utils import cycle_voice
from models import SupContrastiveLoss
import math
from tqdm import tqdm


voice_trans = torch.tensor

vxc_dataset = VoiceDataSet(
    root_path=dataset_config['root_path'],
    voice_frame=dataset_config['voice_frame'],
    voice_ext=dataset_config['voice_ext'],
    voice_transform=voice_trans,
    sample_num=dataset_config['sample_num']
)

voice_loader = DataLoader(vxc_dataset, batch_size=dataset_config['batch_size'], shuffle=True, num_workers=8, collate_fn=dataset_config['collate_fn'], drop_last=True)
voice_iter = cycle_voice(voice_loader)

model = resnet18(pretrained=False, num_classes=512).cuda()
sup_contrastive_loss = SupContrastiveLoss().cuda()
optimizer = Adam(model.parameters(), lr=1e-3)


def adjust_learning_rate(optimizer, epoch, lr=1e-3):
    """Decay the learning rate based on schedule"""
    # cosine lr schedule
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / 400))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # wandb.log({'lr': lr, 'epoch': epoch})

epoch = 1
train_bar = tqdm(range(dataset_config['iters']))
for i in train_bar:
    adjust_learning_rate(optimizer, epoch)
    optimizer.zero_grad()

    voice_batch, label = next(voice_iter)
    l1 = random.randint(400, 600)
    l2 = random.randint(400, 600)
    pt1 = random.randint(0, 100)
    pt2 = random.randint(0, 100)
    voice1, voice2 = voice_batch[:, :, pt1:pt1+l1], voice_batch[:, :, pt2:pt2+l2]
    voice2 = 0.1 * torch.rand_like(voice2) + voice2
    voice1, voice2 = voice1.cuda(), voice2.cuda()
    label = label.cuda()

    f1 = model(voice1.unsqueeze(1))
    f2 = model(voice2.unsqueeze(1))
    f1 = F.normalize(f1, dim=1)
    f2 = F.normalize(f2, dim=1)
    loss = sup_contrastive_loss(f1, f2, label)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    optimizer.step()
    train_bar.set_description(
        f'Train Epoch: [{epoch}], Loss: {loss.item()}')

    if (i+1) % 50 == 0:
        epoch += 1
        torch.save(model.state_dict(), './experiments/voice_res18.pt')

