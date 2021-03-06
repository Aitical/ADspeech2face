from backbones import iresnet34
from dataset import voxceleb1_collate_fn
from models.voice import VoiceStyleNet

exp_name = 'voice_style_encoder'
exp_path = './experiments'
batch_size = 32
sample_num = 4

dataset_config = {
    'root_path': '/home/aitical/data/voxceleb/voxceleb/train',
    # voice dataset
    'voice_ext': 'npy',
    'img_ext': 'jpg',
    'batch_size': 64,
    'voice_frame': [300, 800],
    'num_workers': 8,
    'collate_fn': voxceleb1_collate_fn,
    'sample_num': 2,
    'iters': 50000,
    # test data
    'test_path': '/home/aitical/data4t2/voxceleb/test'
}

voice_encoder = dict(
    model=VoiceStyleNet,
    params=dict(
        input_channel=64,
        output_channel=64,
        channels=[256, 384, 576, 864]
    ),
    pretrained=False,
    model_path='-'
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
    multi_gpu=True,
)
