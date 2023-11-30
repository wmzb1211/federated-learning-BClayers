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
    # w_locals_k_total = torch.stack([torch.stack([torch.flatten(w_locals_k[i])]) for i in range(num_clients)])
    # w_locals_k_total_temp = w_locals_k_total.cpu().numpy()
    # w_locals_k_total_temp = np.array(w_locals_k_total_temp)
    # w_locals_k_total_temp = w_locals_k_total_temp.reshape(w_locals_k_total_temp.shape[0], -1)
    # w_locals_k_total_temp = np.array(w_locals_k_total_temp, dtype=np.double)
    w_locals_k_total = torch.stack([w_locals_k[i].view(-1) for i in range(num_clients)])
    w_locals_k_total_temp = w_locals_k_total.cpu().numpy()
    w_locals_k_total_temp = np.array(w_locals_k_total_temp, dtype=np.double)
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
        # for key, value in temp.items():
        #     if 'weight' in key:
        #         norm_list.append(torch.norm(value).item())
        #     else:
        #         continue
        norm_list.append(torch.norm(temp).item())
    clip_vlaue = np.median(norm_list)
    for b_layer in benign_cluster:
        # for key, value in w_locals_k[b_layer].items():
            # if 'weight' in key:
            #     w_locals_k[b_layer][key] = w_locals_k[b_layer][key] * (clip_vlaue / torch.norm(w_locals_k[b_layer][key]))
            # else:
            #     continue
        w_locals_k[b_layer] = w_locals_k[b_layer] * (clip_vlaue / torch.norm(w_locals_k[b_layer]))

    # 先聚合
    glob_layer = torch.zeros_like(w_locals_k[0])
    for i in benign_cluster:
        temp = w_locals_k[i]
        # for key, value in temp.items():
        #     if key in glob_layer.keys():
        #         glob_layer[key] += value
        #     else:
        #         glob_layer[key] = value
        glob_layer += temp
    # for key, value in glob_layer.items():
    glob_layer = glob_layer / len(benign_cluster)

    # for key, value in w_glob_k.items():
    #     if key in glob_layer.keys():
    #         w_glob_k[key] = glob_layer[key]
    #     else:
    #         continue
    w_glob_k = glob_layer
    # 3.加噪声
    # for key, value in w_glob_k.items():
    #     noise = torch.randn_like(value) * args.noise_scale
    #     w_glob_k[key] = value + noise
    noise = torch.randn_like(w_glob_k) * args.noise_scale
    w_glob_k = w_glob_k + noise
    return w_glob_k




