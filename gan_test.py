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

# initialization
vad_obj = webrtcvad.Vad(2)
mfc_obj = MFCC(nfilt=64, lowerf=20., upperf=7200., samprate=16000, nfft=1024, wlen=0.025)

config_name = sys.argv[1]
command = sys.argv[2]
model_config = importlib.import_module(f'configs.{config_name}')
dataset_config = model_config.dataset_config

model_config.generator['pretrained'] = True

e_net = get_model(model_config.voice_encoder)
g_net = get_model(model_config.generator)

voice_path = os.path.join(dataset_config['test_path'], '*/*/*.wav')
voice_list = glob.glob(voice_path)
for filename in tqdm(voice_list):
    face_image = voice2face(e_net, g_net, filename, vad_obj, mfc_obj, stylegan=True)
    face = face_image[0]
    wav_file_path, wav_file_name = os.path.split(filename)
    face_name = wav_file_name.replace('.wav', f'_{command}.png')
    face_path = wav_file_path.replace('voxceleb', 'voxceleb_face')
    os.makedirs(face_path, exist_ok=True)
    vutils.save_image(face.detach().clamp(-1, 1),
                      os.path.join(face_path, face_name), normalize=True)
