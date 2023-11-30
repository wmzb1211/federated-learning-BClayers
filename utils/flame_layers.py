import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import hdbscan

def flame_layers(w_locals_k: list, w_glob_k: dict, args):
    '''
    :param w_locals_k: 待聚合的本地模型参数的第k层，以state_dict的形式存储
    :param w_glob_k: global model参数的第k层，以state_dict的形式存储
    :param args:
    :return: 聚类+剪枝+加噪声后进行聚合的第k层参数，以state_dict的形式存储
    '''
    # 1.cosine similarity hdbscan clustering
    num_clients = len(w_locals_k)
    clusterer = hdbscan.HDBSCAN(metric="cosine", algorithm='generic', min_cluster_size=num_clients // 2 + 1, min_samples=1, allow_single_cluster=True)
    w_locals_k_total = torch.stack([torch.stack([torch.flatten(w_locals_k[i][k]) for k in w_locals_k[i].keys()]) for i in range(num_clients)])
    clusterer.fit(w_locals_k_total)
    benign_cluster = []
    if clusterer.labels_.max() < 0:
        raise ValueError("No cluster found")
    else:
        for i in range(len(clusterer.labels_)):
            if clusterer.labels_[i] > 0:
                benign_cluster.append(i)
    # 2.剪枝
    # 计算每个benign_cluster的每个参数的平均值
    norm_list = []
    for i in benign_cluster:
        temp = w_locals_k[i]
        for key, value in temp.items():
            if 'weight' in key:
                norm_list.append(torch.norm(value).item())
            else:
                continue
    clip_vlaue = np.median(norm_list)
    for b_layer in benign_cluster:
        for key, value in w_locals_k[b_layer].items():
            if 'weight' in key:
                w_locals_k[b_layer][key] = w_locals_k[b_layer][key] * (clip_vlaue / torch.norm(w_locals_k[b_layer][key]))
            else:
                continue
    # 先聚合
    glob_layer = {}
    for i in benign_cluster:
        temp = w_locals_k[i]
        for key, value in temp.items():
            if key in glob_layer.keys():
                glob_layer[key] += value
            else:
                glob_layer[key] = value
    for key, value in glob_layer.items():
        glob_layer[key] = value / len(benign_cluster)

    for key, value in w_glob_k.items():
        if key in glob_layer.keys():
            w_glob_k[key] = glob_layer[key]
        else:
            continue

    # 3.加噪声
    for key, value in w_glob_k.items():
        noise = torch.randn_like(value) * args.noise_scale
        w_glob_k[key] = value + noise
    return w_glob_k




