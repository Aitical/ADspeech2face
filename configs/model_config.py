from models import VoiceEmbedNet, MixingG
from models.stylegan2 import Discriminator
from backbones import iresnet34
from dataset import voxceleb1_collate_fn


exp_name = 'v',
exp_path = './experiments'

dataset_config = {
    'root_path': '/home/aitical/data/voxceleb/voxceleb/train',
    # voice dataset
    'voice_ext': 'npy',
    'img_ext': 'jpg',
    'batch_size': 4,
    'voice_frame': [300, 800],
    'num_workers': 4,
    'collate_fn': voxceleb1_collate_fn,
    'sample_num': 2,
    'iters': 50000,
    # test data
    'test_path': '/home/aitical/data4t2/voxceleb/test'
}

voice_encoder = dict(
    model=VoiceEmbedNet,
    params=dict(
        input_channel=64,
        channels=[256, 384, 576, 864],
        output_channel=64
    ),
    pretrained=True,
    model_path='pretrained_models/speech2face_model/voice_embedding.pth'
)

generator = dict(
    model=MixingG,
    params=dict(
        input_channel=64,
        channels=[1024, 512, 256, 128, 64],
        output_channel=3
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
        betas=(0.5,0.999)
    ),
    multi_gpu=True,
)
