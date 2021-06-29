from models import resnet50
from dataset import get_dataset
from utils import cycle_voice
from torch.utils.data import DataLoader
from configs.nips import DATASET_PARAMETERS
import torch
# dataset and dataloader
print('Parsing your dataset...')
voice_list, face_list, id_class_num = get_dataset(DATASET_PARAMETERS)
print('Preparing the datasets...')
voice_dataset = DATASET_PARAMETERS['voice_dataset'](voice_list,
                               DATASET_PARAMETERS['nframe_range'])

print('Preparing the dataloaders...')
collate_fn = DATASET_PARAMETERS['collate_fn'](DATASET_PARAMETERS['nframe_range'])
voice_loader = DataLoader(voice_dataset, shuffle=True, drop_last=True,
                          batch_size=DATASET_PARAMETERS['batch_size'],
                          num_workers=DATASET_PARAMETERS['workers_num'],
                          collate_fn=collate_fn)

voice_iterator = iter(cycle_voice(voice_loader))
m = resnet50(pretrained=False, num_classes=1024)

for i in voice_iterator:
    l, c = torch.unique(i[1], return_counts=True)
    print(l, c)