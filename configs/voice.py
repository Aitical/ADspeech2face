from dataset import voice_collate_fn


dataset_config = {
    'root_path': './experiments/dataset/train',
    # voice dataset
    'voice_ext': 'npy',
    'batch_size': 64,
    'voice_frame': [300, 800],
    'num_workers': 4,
    'collate_fn': voice_collate_fn,
    'sample_num': 2,
    'iters': 20000,
    # test data
    'test_path': './experiments/dataset/test',
    'save_path': './experiments/voice.pt'
}
