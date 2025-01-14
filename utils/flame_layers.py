import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import hdbscan
import sklearn
def flame_layers(w_locals_k_: list, args):
    '''
    :param w_locals_k: 待聚合的本地模型参数的第k层，以state_dict的形式存储
    :param w_glob_k: global model参数的第k层，以state_dict的形式存储
    :param args:
    :return: 聚类+剪枝+加噪声后进行聚合的第k层参数，以state_dict的形式存储
    '''
    # 1.cosine similarity hdbscan clustering
    # cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    # cos_list = []
    num_clients = len(w_locals_k_)
    w_locals_k = []
    for i in range(num_clients):
        w_locals_k.append(w_locals_k_[i].reshape(-1).cpu())
    clusterer = hdbscan.HDBSCAN(metric="cosine", algorithm='generic', min_cluster_size=num_clients // 2 + 1, max_cluster_size= int(num_clients * 3 / 4)  ,min_samples=1, allow_single_cluster=True)
    # for i in range(num_clients):
    #     cos_i = []
    #     for j in range(num_clients):
    #         cos_ij = cos(w_locals_k[i], w_locals_k[j])
    #         cos_i.append(cos_ij.item())
    #     cos_list.append(cos_i)
    # clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients // 2 + 1, max_cluster_size= int(num_clients * 3 / 4)  ,min_samples=1, allow_single_cluster=True)
    w_locals_k_total = torch.stack(w_locals_k)
    small_value = 0.00001
    w_locals_k_total = w_locals_k_total + small_value
    w_locals_k_total_temp = w_locals_k_total.double()
    print("w_locals_k_total_temp.shape: ", w_locals_k_total_temp.shape)
    # w_locals_k_total_temp = np.array(w_locals_k_total_temp, dtype=np.double)
    # print('cos_list: ', cos_list)
    # clusterer.fit(cos_list)
    clusterer.fit(w_locals_k_total_temp)
    print("clusterer.labels_: ", clusterer.labels_)
    benign_cluster = []
    if clusterer.labels_.max() < 0:
        raise ValueError("No cluster found")
    else:
        for i in range(len(clusterer.labels_)):
            if clusterer.labels_[i] >= 0:
                benign_cluster.append(i)
    print("benign_cluster: ", benign_cluster)
    # 2.剪枝
    # 计算每个benign_cluster的每个参数的平均值
    norm_list = []
    for i in benign_cluster:
        temp = w_locals_k_[i]
        norm_list.append(torch.norm(temp).item())
    clip_vlaue = np.median(norm_list)
    for b_layer in benign_cluster:
        w_locals_k_[b_layer] = w_locals_k_[b_layer] * (clip_vlaue / torch.norm(w_locals_k_[b_layer]))

    # 先聚合
    glob_layer = w_locals_k_[benign_cluster[0]]
    for i in benign_cluster[1:]:
        temp = w_locals_k_[i]
        glob_layer += temp
    glob_layer = glob_layer / len(benign_cluster)
    w_glob_k = glob_layer
    # 3.加噪声
    noise = torch.randn_like(w_glob_k) * args.noise_scale
    w_glob_k = w_glob_k + noise
    return w_glob_k




