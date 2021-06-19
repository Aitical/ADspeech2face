from torch.utils.data import Dataset
import pathlib
import os
import random
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
import torch


class VoxCeleb1DataSet(Dataset):
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

        for id_folder in folders:

            id_folder_path = os.path.join(root, id_folder)
            id_folder_path = pathlib.Path(id_folder_path)
            voices = [i for i in id_folder_path.glob(f'voice/*.{voice_ext}')]
            imgs = [i for i in id_folder_path.glob(f'face/*.{img_ext}')]

            id_message = dict(
                voice_count=len(voices),
                img_count=len(imgs),
                voices=voices,
                imgs=imgs,
                id=id_folder,
                label=id2label[id_folder]
            )
            data.append(id_message)
        self.data = data
        self.length = len(self.data)
        self.num_classes = self.length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        #  print(index)
        data_message = self.data[index]
        voice_samples = []
        img_samples = []
        img_lr_samples = []
        labels = [data_message['label']]*self.sample_num
        for c in range(self.sample_num):
            # print(data_message['id'], data_message['voice_count'])
            voice_sample = data_message['voices'][random.randint(0, data_message['voice_count']-1)]
            img_sample = data_message['imgs'][random.randint(0, data_message['img_count']-1)]

            voice_data = np.load(voice_sample).T.astype('float32')
            # print(voice_data.shape)
            assert self.voice_frame[1] <= voice_data.shape[1]
            frame_length = self.voice_frame[1]  # random.randint(self.voice_frame[0], self.voice_frame[1])
            pt = np.random.randint(voice_data.shape[1] - frame_length + 1)
            voice_data = voice_data[:, pt:pt + frame_length]

            img_data = Image.open(img_sample).convert('RGB')
            img_lr_data = img_data.resize((16, 16))

            if self.voice_transform is not None:
                voice_data = self.voice_transform(voice_data)

            voice_samples.append(voice_data)
            if self.img_transform is not None:
                img_data = self.img_transform(img_data)
                img_lr_data = self.img_transform(img_lr_data)

            img_samples.append(img_data)
            img_lr_samples.append(img_lr_data)
        # if self.img_transform is not None:
        #     img_samples = torch.stack(img_samples)
        # voice_samples = np.array(voice_samples)
        # labels = torch.tensor(labels).long()
        return img_samples, voice_samples, labels, img_lr_samples


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
    mydataset = VoxCeleb1DataSet(root_path='/home/aitical/data/voxceleb/voxceleb/train', sample_num=4, img_transform=face_transform, voice_transform=voice_trans)

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
    dataloader = DataLoader(mydataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=coll_fn)

    print(len(mydataset))
    for i, k, c in dataloader:
        # print(type(i), type(k), type(c))
        print(i.shape, k.shape, c.shape)
        break