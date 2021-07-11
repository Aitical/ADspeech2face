import torch
from torchvision import transforms
import configs.criteria
from dataset import VoxCeleb1DataSet
from torch.utils.data import DataLoader
from dataset import cycle_data
from configs import model_config
from dataset import get_dataset
from utils import cycle_voice, cycle_face

def get_vxc_data_iter(dataset_config):
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

    print(len(vxc_dataset))
    train_loader = DataLoader(
        vxc_dataset,
        shuffle=True,
        batch_size=dataset_config['batch_size'],
        num_workers=dataset_config['num_workers'],
        collate_fn=dataset_config['collate_fn'],
        drop_last=True
    )

    data_iter = cycle_data(train_loader)
    return data_iter


def get_nips_data_iter(dataset_config):
    print('Parsing your dataset...')
    voice_list, face_list, id_class_num = get_dataset(dataset_config)
    voice_dataset = dataset_config['voice_dataset'](voice_list,
                                                    dataset_config['nframe_range'])
    face_dataset = dataset_config['face_dataset'](face_list)

    print('Preparing the dataloaders...')
    collate_fn = dataset_config['collate_fn'](dataset_config['nframe_range'])

    voice_loader = DataLoader(voice_dataset, shuffle=True, drop_last=True,
                                  batch_size=dataset_config['batch_size'],
                                  num_workers=dataset_config['workers_num'],
                                  collate_fn=collate_fn)

    face_loader = DataLoader(face_dataset, shuffle=True, drop_last=True,
                             batch_size=dataset_config['batch_size'],
                             num_workers=dataset_config['workers_num'])

    voice_iterator = iter(cycle_voice(voice_loader))
    face_iterator = iter(cycle_face(face_loader))
    return voice_iterator, face_iterator


def get_model(config_dict, multi_gpu=True):
    model = config_dict['model'](**config_dict['params'])

    if config_dict['pretrained'] and config_dict['model_path'] != '-':
        miss = model.load_state_dict(torch.load(config_dict['model_path'], map_location='cpu'))
        print(miss)
        model.eval()

    elif config_dict['pretrained']:
        for param in model.parameters():
            param.requires_grad = False

    if multi_gpu:
        model = torch.nn.DataParallel(model)
        model.cuda()
    else:
        model.cuda()
    return model


def get_edsr():
    from edsr.model import Model
    model = Model(configs.criteria.model_paths['edsr'])
    model = model.model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model = torch.nn.DataParallel(model)
    model.cuda()
    return model


def get_arcface():
    arcface = get_model(model_config.arcface)
    return arcface