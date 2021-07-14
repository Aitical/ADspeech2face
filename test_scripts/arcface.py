import torch
from PIL import Image
import torchvision.transforms as transforms
from configs import model_config
from parse_config import get_model


model = get_model(model_config.arcface)

img_trans = transforms.Compose([
    transforms.Resize(size=(112, 112)),
    transforms.ToTensor(),
    # transforms.Normalize(0)
])

img1 = Image.open('./test_scripts/liu1.jpg')
img1 = img_trans(img1)

img2 = Image.open('./test_scripts/liu2.jpg')
img2 = img_trans(img2)

img = torch.cat([img1.unsqueeze(0), img2.unsqueeze(0)], dim=0)
f = model(img)
f1, f2 = f[0], f[1]
print(f1.shape, f2.shape)
f1 = torch.nn.functional.normalize(f1.unsqueeze(0), dim=1)
f2 = torch.nn.functional.normalize(f2.unsqueeze(0), dim=1)

print(f1.mm(f2.t()))