import os
import glob
import torch
import torchvision.utils as vutils
import webrtcvad
from mfcc import MFCC
from utils import voice2face_stylegan
from tqdm import tqdm
import sys
from parse_config import get_model
import importlib
from models.stylegan2_pytorch import ModelLoader
from configs.criteria import model_paths
from models.voice import StyleMapping

# initialization
vad_obj = webrtcvad.Vad(2)
mfc_obj = MFCC(nfilt=64, lowerf=20., upperf=7200., samprate=16000, nfft=1024, wlen=0.025)

config_name = sys.argv[1]
command = sys.argv[2]
model_config = importlib.import_module(f'configs.{config_name}')
dataset_config = model_config.dataset_config

# model_config.generator['pretrained'] = True
experiment_name = model_config.exp_name
experiment_path = model_config.exp_path
save_path = os.path.join(experiment_path, experiment_name)

e_net = get_model(model_config.voice_encoder)

g_loader = ModelLoader(
    base_dir=model_paths['stylegan2-pytorch'],   # path to where you invoked the command line tool
    name='default'                   # the project name, defaults to 'default'
)
g_net = g_loader.model.GAN.GE
num_layers = g_loader.model.GAN.GE.num_layers
img_size = g_loader.model.GAN.GE.image_size

style_mapping = StyleMapping(style_dim=512, style_heads=num_layers, mlp_dim=512)
style_mapping.load_state_dict(torch.load(os.path.join(save_path, 'stylemapping.pt'), map_location='cpu'))
style_mapping.cuda()


voice_path = os.path.join(dataset_config['test_path'], '*/*/*.wav')
voice_list = glob.glob(voice_path)
for filename in tqdm(voice_list):
    face_image = voice2face_stylegan(e_net, style_mapping, g_net, filename, vad_obj, mfc_obj, stylegan=True)
    face = face_image[0]
    wav_file_path, wav_file_name = os.path.split(filename)
    face_name = wav_file_name.replace('.wav', f'_{command}.png')
    face_path = wav_file_path.replace('voxceleb', 'voxceleb_face')
    os.makedirs(face_path, exist_ok=True)
    vutils.save_image(face.detach().clamp(-1, 1),
                      os.path.join(face_path, face_name), normalize=True)
