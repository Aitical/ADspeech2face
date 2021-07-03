import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLRLoss(nn.Module):
    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()

        self.T = temperature
        self.ce = nn.CrossEntropyLoss()
        self.norm = nn.functional.normalize
        self.softmax = nn.functional.softmax
        self.cosine = nn.CosineSimilarity(dim=-1)
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.mse = nn.MSELoss()

    def _get_correlated_mask(self, batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = diag + l1 + l2
        # mask = (1 - mask).type(torch.bool)
        return mask

    def forward(self, f1, f2):
        batch_size = f1.shape[0]
        # f = torch.cat([f1, f2], dim=0)

        sim_matrix = self.cosine(f1.unsqueeze(1), f2.unsqueeze(0)) / self.T
        label = torch.arange(0, batch_size, device=sim_matrix.device)

        loss = self.ce(sim_matrix, label) + self.ce(sim_matrix.t(), label)
        return loss * 0.5


class SupContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.t = temperature

    def forward(self, x1, x2, label):
        if len(x1.shape) > 2:
            x1 = x1.squeeze()
            x2 = x2.squeeze()
            assert len(x1.shape)==2
        # x1 = x1.view(x1.size()[0], -1)
        # x2 = x2.view(x2.size()[0], -1)
        # BxB
        sim = torch.exp(x1.mm(x2.t()) / self.t)
        # print(sim.shape)
        label = label.reshape(-1, 1)
        # BxB
        label_matrix = label.eq(label.t()).float()
        reg_value = 1 / label_matrix.sum(dim=1, keepdim=True)
        logits = -torch.log(sim/sim.sum(dim=1, keepdim=True)) * label_matrix * reg_value
        loss = torch.sum(logits, dim=1).mean()
        return loss

def gen_hinge_loss(fake, real):
    return fake.mean()

def hinge_loss(real, fake):
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()
