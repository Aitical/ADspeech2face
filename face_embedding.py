import argparse

import cv2
import numpy as np
import torch
import os

import torchvision.transforms as transforms
from backbones import get_model
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pathlib

class VGGFace(Dataset):
    def __init__(self, root, transforms=None):
        super().__init__()
        self.root_path = root
        self.transforms = transforms
        names = os.listdir(root)
        self.name2label = name2id
        name_contain = []
        index_contain = []
        for n in name2id:
            if n in names:
                name_contain.append(n)
                index_contain.append(name2id[n])
        self.name2label = dict(zip(name_contain, index_contain))
        self.names = name_contain
        print(len(self.names))
        file_path = []
        targets = []
        for name in self.names:
            folder_path = os.path.join(self.root_path, name)
            img_files = os.listdir(folder_path)
            for img_name in img_files:
                file_path.append(os.path.join(folder_path, img_name))
                targets.append(self.name2label[name])

        self.data = file_path
        self.targets = targets
        assert len(self.data) == len(self.targets)
        self.length = len(targets)

    def __getitem__(self, item):
        img = Image.open(self.data[item]).convert('RGB')
        img = img.resize((112, 112))
        label = self.targets[item]

        if self.transforms is not None:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return self.length



class CelebFace(Dataset):
    def __init__(self, root, index_list, tag='', transforms=None):
        super().__init__()
        self.root_path = root
        self.transforms = transforms
        # names = os.listdir(root)
        self.names =  os.listdir(root)
        # self.name2label = dict(zip(names, list(range(len(names)))))
        file_path = []
        targets = []
        for name in self.names:
            id_path = os.path.join(root, name)
            current_id_imgs = [str(i) for i in pathlib.Path(id_path).glob(f'*/*_{tag}.png')]
            file_path.extend(current_id_imgs)
            targets.extend([name for i in range(len(current_id_imgs))])

        self.data = file_path
        self.targets = targets
        assert len(self.data) == len(self.targets)
        self.length = len(targets)

    def __getitem__(self, item):
        img = Image.open(self.data[item]).convert('RGB')
        img = img.resize((112, 112))
        label = self.targets[item]

        if self.transforms is not None:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return self.length



@torch.no_grad()
def inference_vggface(weight, name, dataloader, save_path):
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.cuda()
    net.eval()
    features = []
    targets = []
    for img, labels in tqdm(dataloader):
        img = img.cuda()
        feat = net(img).cpu()
        features.append(feat)
        targets.extend(labels)
    features = torch.cat(features, dim=0)
    results = {'features': features, 'targets': targets}
    torch.save(results, save_path)
    return results

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('--img', type=str, default=None)
    parser.add_argument('--anchor_embedding', type=str)
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--save_path', type=str,)
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()


    test_trans = transforms.Compose([transforms.ToTensor()])
    raw_embedding = torch.load(args.anchor_embedding)
    index_list = np.unique(raw_embedding['targets'])

    celeb_face = CelebFace(args.img_path, index_list=index_list, tag=args.tag, transforms=test_trans)
    print(len(celeb_face))
    # vgg_face = VGGFace(root=args.img_path, transforms=test_trans)
    face_loader = DataLoader(celeb_face, batch_size=64, shuffle=False, num_workers=16)

    inference_vggface(args.weight, args.network, face_loader, args.save_path)
