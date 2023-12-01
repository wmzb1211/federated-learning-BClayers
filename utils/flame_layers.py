import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import hdbscan
import sklearn
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
    w_locals_k_total = torch.stack(w_locals_k)
    w_locals_k_total_temp = w_locals_k_total.double()
    print("w_locals_k_total_temp.shape: ", w_locals_k_total_temp.shape)
    # w_locals_k_total_temp = np.array(w_locals_k_total_temp, dtype=np.double)
    clusterer.fit(w_locals_k_total_temp)
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
        norm_list.append(torch.norm(temp).item())
    clip_vlaue = np.median(norm_list)
    for b_layer in benign_cluster:
        w_locals_k[b_layer] = w_locals_k[b_layer] * (clip_vlaue / torch.norm(w_locals_k[b_layer]))

    # 先聚合
    glob_layer = torch.zeros_like(w_locals_k[0])
    for i in benign_cluster:
        temp = w_locals_k[i]
        glob_layer += temp
    glob_layer = glob_layer / len(benign_cluster)
    w_glob_k = glob_layer
    # 3.加噪声
    noise = torch.randn_like(w_glob_k) * args.noise_scale
    w_glob_k = w_glob_k + noise
    return w_glob_k




