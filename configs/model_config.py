from models import VoiceEmbedNet, MixingG
from models.stylegan2 import Discriminator
from backbones import iresnet34


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
    # model_path=f'./experiments/{experiment_name}/generator.pth',
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
    exp_name='v',
    exp_path='./experiments'
)
