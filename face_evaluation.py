from mmap import MADV_RANDOM
import torch
from collections import defaultdict
from tqdm import tqdm
import sys

def face_evaluation(embd1, embd2):
    res1 = torch.load(embd1)
    res2 = torch.load(embd2)

    anchor_dict = defaultdict(list)
    for i in range(len(res1['targets'])):
        index = res1['targets'][i]
        # print(anchor_dict[index], index)
        anchor_dict[index].append(res1['features'][i])

    test_dict = defaultdict(list)
    for i in range(len(res2['targets'])):
        index = res2['targets'][i]
        test_dict[index].append(res2['features'][i])


    res = {}
    for k in test_dict:
        test_embed = torch.stack(test_dict[k], dim=0)

        anchor_embed = torch.stack(anchor_dict[k], dim=0)
        # print(test_embed.shape, anchor_embed.shape)
        # MxD DxN
        test_embed = torch.nn.functional.normalize(test_embed, dim=1)
        anchor_embed = torch.nn.functional.normalize(anchor_embed, dim=1)
        sim = test_embed.mm(anchor_embed.t())
        res[k] = [sim.mean(), sim.max(), sim.min()]
        # print(res[k])
    return res


if __name__ == '__main__':
    anchor_path, test_path = sys.argv[1], sys.argv[2]
    res = face_evaluation(anchor_path, test_path)
    metric = torch.tensor(list(res.values()))
    # print(metric.shape)
    print(metric.mean(dim=0))
