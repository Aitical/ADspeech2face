import os
import glob
import torch
import torchvision.utils as vutils
import webrtcvad

from mfcc import MFCC
from configs.config254 import dataset_config, NETWORKS_PARAMETERS
from models import get_network
# from dataset import VoxCeleb1DataSet
from utils import voice2face
from tqdm import tqdm
import sys
from models.stylegan2 import Generator


# initialization
vad_obj = webrtcvad.Vad(2)
mfc_obj = MFCC(nfilt=64, lowerf=20., upperf=7200., samprate=16000, nfft=1024, wlen=0.025)
e_net, _ = get_network('e', NETWORKS_PARAMETERS, train=False)
# g_net, _ = get_network('g', NETWORKS_PARAMETERS, train=False)
g_net = Generator(128, 64, 4)
#g_net = torch.nn.DataParallel(g_net)
miss = g_net.load_state_dict(torch.load(NETWORKS_PARAMETERS['g']['model_path']))
# torch.save(g_net.module.state_dict(), NETWORKS_PARAMETERS['g']['model_path'])
print(miss, 'ok')
g_net.eval()
g_net.cuda()

command = sys.argv[1]

voice_path = os.path.join(dataset_config['test_path'], '*/*/*.wav')
voice_list = glob.glob(voice_path)
for filename in tqdm(voice_list):
    face_image = voice2face(e_net, g_net, filename, vad_obj, mfc_obj,
                            NETWORKS_PARAMETERS['GPU'], stylegan=True)
    face = face_image[0]
    wav_file_path, wav_file_name = os.path.split(filename)
    face_name = wav_file_name.replace('.wav', f'_{command}.png')
    face_path = wav_file_path.replace('voxceleb', 'voxceleb_face')
    os.makedirs(face_path, exist_ok=True)
    vutils.save_image(face.detach().clamp(-1, 1),
                      os.path.join(face_path, face_name), normalize=True)
