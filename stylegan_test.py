import os
import glob
import torch
import torchvision.utils as vutils
import webrtcvad
from mfcc import MFCC
from utils import voice2face
from tqdm import tqdm
import sys
from parse_config import get_model
import importlib
from models.stylegan2_pytorch import ModelLoader
from configs.criteria import model_paths
from models.stylegan import VoiceEmbedNet
from models.stylegan2 import Generator


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


g_ema_ = Generator(64, 8, 512, 2).cuda()

style_mapping = VoiceEmbedNet(g_ema_).cuda()
ckpt = torch.load('/home/aitical/Documents/paper_with_code/speech2face/adspeech2face/experiments/voice_style_encoder/embedding_style.pt')
style_mapping.load_state_dict(ckpt)
style_mapping.eval()

voice_path = os.path.join(dataset_config['test_path'], '*/*/*.wav')
voice_list = glob.glob(voice_path)
for filename in tqdm(voice_list):
    face_image = voice2face(style_mapping, g_net, filename, vad_obj, mfc_obj, stylegan=True)
    face = face_image[0]
    wav_file_path, wav_file_name = os.path.split(filename)
    face_name = wav_file_name.replace('.wav', f'_{command}.png')
    face_path = wav_file_path.replace('voxceleb', 'voxceleb_face')
    os.makedirs(face_path, exist_ok=True)
    vutils.save_image(face.detach().clamp(-1, 1),
                      os.path.join(face_path, face_name), normalize=True)
