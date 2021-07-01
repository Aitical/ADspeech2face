import string
from dataset import VoiceDataset, FaceDataset
from utils import get_collate_fn

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
    'batch_size': 1024,
    'nframe_range': [300, 800],
    'workers_num': 1,
    'collate_fn': get_collate_fn,

    # test data
    'test_data': 'data/test_data/'
}

