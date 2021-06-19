from network import VoiceEmbedNet, Generator, FaceEmbedNet, Classifier, BasicGenerator, BSEGenerator, LightG
from dataset import voxceleb1_collate_fn
from backbones import iresnet34


dataset_config = {
    'root_path': '/home/aitical/data/voxceleb/voxceleb/train',
    # voice dataset
    'voice_ext': 'npy',
    'img_ext': 'jpg',
    'batch_size': 32,
    'voice_frame': [300, 600],
    'num_workers': 4,
    'collate_fn': voxceleb1_collate_fn,
    'sample_num': 4,
    # test data
    'test_path': '/home/aitical/data/voxceleb/voxceleb/test'
}


experiment_name = 'BSE_32_4'
experiment_path = './experiments'

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
        'network': BSEGenerator,
        'input_channel': 64,
        'channels': [1024, 512, 256, 128, 64],  # channels for deconvolutional layers
        'output_channel': 3,  # images with RGB channels
        'model_path': f'./experiments/{experiment_name}/generator_l1_edsr_16_64.pth',
    },
    # FACE EMBEDDING NETWORK (f)
    'f': {
        'network': FaceEmbedNet,
        'input_channel': 3,
        'channels': [32, 64, 128, 256, 512],
        'output_channel': 64,
        'model_path': f'./experiments/{experiment_name}/face_embedding.pth',
    },
    # DISCRIMINATOR (d)
    'd': {
        'network': Classifier,  # Discrminator is a special Classifier with 1 subject
        'input_channel': 64,
        'channels': [],
        'output_channel': 1,
        'model_path': f'./experiments/{experiment_name}/discriminator.pth',
    },
    # CLASSIFIER (c)
    'c': {
        'network': Classifier,
        'input_channel': 64,
        'channels': [],
        'output_channel': -1,  # This parameter is depended on the dataset we used
        'model_path': f'./experiments/{experiment_name}/classifier.pth',
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
