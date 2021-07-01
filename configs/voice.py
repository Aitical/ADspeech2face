from dataset import voice_collate_fn


dataset_config = {
    'root_path': '/home/aitical/data/voxceleb/voxceleb/train',
    # voice dataset
    'voice_ext': 'npy',
    'batch_size': 64,
    'voice_frame': [300, 800],
    'num_workers': 4,
    'collate_fn': voice_collate_fn,
    'sample_num': 2,
    'iters': 50000,
    # test data
    'test_path': '/home/aitical/data4t2/voxceleb/test',
    'save_path': './experiments/voice.pt'
}
