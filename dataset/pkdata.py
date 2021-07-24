from torch.utils.data import Dataset
import pathlib
import os
import random
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
import torch


class PKVoxCeleb1DataSet(Dataset):
    def __init__(self, root_path, sample_num, voice_frame=[300, 800], voice_transform=None, img_transform=None,  voice_ext='npy', img_ext='jpg'):
        root = pathlib.Path(root_path)
        data = []
        folders = os.listdir(root)
        id_index = list(range(len(folders)))
        id2label = dict(zip(folders, id_index))
        self.id2label = id2label
        self.id_name = folders
        self.sample_num = sample_num
        self.voice_transform = voice_transform
        self.img_transform = img_transform

        self.voice_frame = voice_frame

        img2id = []

        for id_folder in folders:

            id_folder_path = os.path.join(root, id_folder)
            id_folder_path = pathlib.Path(id_folder_path)
            voices = [i for i in id_folder_path.glob(f'voice/*.{voice_ext}')]
            imgs = [i for i in id_folder_path.glob(f'face/*.{img_ext}')]
            img2id.extend([(i, id2label[id_folder]) for i in id_folder_path.glob(f'face/*.{img_ext}')])

            id_message = dict(
                voice_count=len(voices),
                img_count=len(imgs),
                voices=voices,
                imgs=imgs,
                id=id_folder,
                label=id2label[id_folder]
            )
            data.append(id_message)
        self.img_path = img2id
        self.data = data
        self.length = len(self.img_path)
        self.num_classes = self.length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        #  print(index)
        img_path, id_index = self.img_path[index]
        data_message = self.data[id_index]

        img_data = Image.open(img_path).convert('RGB')
        if self.img_transform is not None:
            img_data = self.img_transform(img_data)

        labels = data_message['label']
        voice_sample = data_message['voices'][random.randint(0, data_message['voice_count']-1)]

        voice_data = np.load(voice_sample).T.astype('float32')
        assert self.voice_frame[1] <= voice_data.shape[1]
        frame_length = self.voice_frame[1]  # random.randint(self.voice_frame[0], self.voice_frame[1])
        pt = np.random.randint(voice_data.shape[1] - frame_length + 1)
        voice_data = voice_data[:, pt:pt + frame_length]

        if self.voice_transform is not None:
            voice_data = self.voice_transform(voice_data)

        return img_data, voice_data, labels


def voxceleb1_collate_fn(batch):
    # img, voice, labels = batch
    img = []
    voice = []
    label = []
    img_lr = []
    random.shuffle(batch)

    for i in batch:
        img.extend(i[0])
        voice.extend(i[1])
        label.extend(i[2])
        img_lr.extend(i[3])

    img = torch.stack(img)
    img_lr = torch.stack(img_lr)
    voice = torch.stack(voice)
    label = torch.tensor(label).long()
    # print(img.shape)
    return img, voice, label, img_lr


def voice_collate_fn(batch):
    # img, voice, labels = batch
    # img = []
    voice = []
    label = []
    # img_lr = []
    random.shuffle(batch)

    for i in batch:
        # img.extend(i[0])
        voice.extend(i[0])
        label.extend(i[1])
        # img_lr.extend(i[3])


    voice = torch.stack(voice)
    label = torch.tensor(label).long()
    # print(img.shape)
    return voice, label

def cycle_data(dataloader):
    while True:
        for img, voice, label, lr in dataloader:
            yield img, voice, label, lr


if __name__ == '__main__':
    face_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)]
    )
    voice_trans = transforms.Compose(
        [
            torch.tensor
        ]
    )
    mydataset = PKVoxCeleb1DataSet(root_path='/home/aitical/data/voxceleb/voxceleb/train', sample_num=4, img_transform=face_transform, voice_transform=voice_trans)

    def coll_fn(batch):
        # img, voice, labels = batch
        img = []
        voice = []
        label = []
        for i in batch:
            img.extend(i[0])
            voice.extend(i[1])
            label.extend(i[2])

        img = torch.stack(img)
        voice = torch.stack(voice)
        label = torch.tensor(label).long()
        # print(img.shape)
        return img, voice, label

    from torch.utils.data import DataLoader
    dataloader = DataLoader(mydataset, batch_size=32, shuffle=True, num_workers=4)

    print(len(mydataset))
    for i, k, c in dataloader:
        # print(type(i), type(k), type(c))
        print(i.shape, k.shape, c.shape)
        break