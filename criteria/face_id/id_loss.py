import torch
from torch import nn
import torch.nn.functional as F
from configs.criteria import model_paths
from criteria.face_id.model_irse import Backbone


class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(model_paths['ir_se50']))
        # self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

        for param in self.facenet.parameters():
            param.requires_grad = False

    def extract_feats(self, x, mode='center'):
        if mode == 'center':
            x = x[:, :, 8:120, 8:120]  # Crop interesting region
        elif mode == 'resize':
            x = F.interpolate(x, size=112)
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y):
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 1 - torch.sum(y_hat_feats * y_feats, dim=1).mean()
        return loss
