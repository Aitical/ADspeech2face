import torch
from torchvision.utils import save_image
from models.stylegan2_pytorch import ModelLoader


loader = ModelLoader(
    base_dir = '/home/aitical/Documents/paper_with_code/speech2face/stylegan2/',   # path to where you invoked the command line tool
    name = 'default'                   # the project name, defaults to 'default'
)

noise   = torch.randn(3, 512).cuda() # noise
styles  = loader.noise_to_styles(noise, trunc_psi = 0.7)  # pass through mapping network
print(styles.shape)
images  = loader.styles_to_images(styles) # call the generator on intermediate style vectors

save_image(images, './sample.jpg')

g_net = loader.model.GAN.GE
for p in g_net.parameters():
    if p.requires_grad:
        print(p, p.requires_grad)