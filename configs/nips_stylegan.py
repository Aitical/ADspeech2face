from models.voice import ResNetSE34
from models.stylegan2 import Generator, Discriminator
from models.stylegan2 import Discriminator
from backbones import iresnet34
from dataset import VoiceDataset, FaceDataset
import string
from utils import get_collate_fn

exp_name = 'nips_stylegan'
exp_path = './experiments'


dataset_config = {
    # meta data provided by voxceleb1 dataset
    'meta_file': 'data/vox1_meta.csv',

    # voice dataset
    'voice_dir': '/home/aitical/data4t/voxceleb/voxceleb/fbank',
    'voice_ext': 'npy',

    # face dataset
    'face_dir': '/home/aitical/data4t/voxceleb/voxceleb/VGG_ALL_FRONTAL',
    'face_ext': '.jpg',

    # train data includes the identities
    # whose names start with the characters of 'FGH...XYZ'
    'split': string.ascii_uppercase[5:],

    # dataloader
    'voice_dataset': VoiceDataset,
    'face_dataset': FaceDataset,
    'batch_size': 32,
    'nframe_range': [300, 800],
    'workers_num': 8,
    'collate_fn': get_collate_fn,

    # test data
    'test_data': '/home/aitical/data4t2/voxceleb/test'
}


voice_encoder = dict(
    model=ResNetSE34,
    params=dict(
    ),
    pretrained=True,
    model_path='-'
)

generator = dict(
    model=Generator,
    params=dict(
        size=128,
        style_dim=512,
        n_mlp=8,
    ),
    pretrained=False,
    model_path=f'./{exp_path}/{exp_name}/generator.pth',
)

discriminator = dict(
    model=Discriminator,
    params=dict(
        size=128
    ),
    pretrained=False,
)

arcface = dict(
    model=iresnet34,
    params=dict(
        fp16=False,
    ),
    pretrained=True,
    model_path='./pretrained_models/arc_face_model/backbone.pth'
)


training_config = dict(
    optimizer=dict(
        lr=2e-3,
        betas=(0.5, 0.999)
    ),
    multi_gpu=False,
)
