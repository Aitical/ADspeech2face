import string
from dataset import VoiceDataset, FaceDataset
from network import VoiceEmbedNet, Generator, FaceEmbedNet, Classifier, BasicGenerator, BSEGenerator, LightG
from utils import get_collate_fn
from backbones import iresnet34



DATASET_PARAMETERS = {
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
    'batch_size': 128,
    'nframe_range': [300, 800],
    'workers_num': 1,
    'collate_fn': get_collate_fn,

    # test data
    'test_data': '/home/aitical/data4t2/voxceleb/test'
}


NETWORKS_PARAMETERS = {
    # VOICE EMBEDDING NETWORK (e)
    'e': {
        'network': VoiceEmbedNet,
        'input_channel': 64,
        'channels': [256, 384, 576, 864],
        'output_channel': 64,  # the embedding dimension
        'model_path': 'pretrained_models/speech2face_model/voice_embedding.pth',
    },
    # GENERATOR (g)
    'g': {
        'network': LightG,
        'input_channel': 64,
        'channels': [1024, 512, 256, 128, 64],  # channels for deconvolutional layers
        'output_channel': 3,  # images with RGB channels
        'model_path': './experiments/arcface/generator_l1_edsr_16_64.pth',
    },
    # FACE EMBEDDING NETWORK (f)
    'f': {
        'network': FaceEmbedNet,
        'input_channel': 3,
        'channels': [32, 64, 128, 256, 512],
        'output_channel': 64,
        'model_path': './experiments/arcface/face_embedding.pth',
    },
    # DISCRIMINATOR (d)
    'd': {
        'network': Classifier,  # Discrminator is a special Classifier with 1 subject
        'input_channel': 64,
        'channels': [],
        'output_channel': 1,
        'model_path': './experiments/arcface/discriminator.pth',
    },
    # CLASSIFIER (c)
    'c': {
        'network': Classifier,
        'input_channel': 64,
        'channels': [],
        'output_channel': -1,  # This parameter is depended on the dataset we used
        'model_path': './experiments/arcface/classifier.pth',
    },

    'arcface': {
        'network': iresnet34,
        'fp16': False,
        'model_path': './pretrained_models/arc_face_model/backbone.pth'
    },

    # OPTIMIZER PARAMETERS 
    'lr': 0.0002,
    'beta1': 0.5,
    'beta2': 0.999,

    # MODE, use GPU or not
    'GPU': True,
}
