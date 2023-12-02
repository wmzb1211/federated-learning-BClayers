import os
import torch
import numpy as np
from utils.flame_layers import flame_layers

# def fedavg_layers(w_locals_k, net_glob_k):
#     '''
#     :param w_locals: list of state_dict, 代表各个client的本地模型参数
#     :param net_glob: model, 代表global model
#     :return:
#     '''
#     for key, value in net_glob_k.items():
#         for w_local in w_locals_k:
#             value += w_local[key] / len(w_locals_k)
#     return net_glob_k
def fedavg_layers(w_locals_k, net_glob_k):
    '''
    :param w_locals: list of state_dict, 代表各个client的本地模型参数
    :param net_glob: model, 代表global model
    :return:
    '''
    value = net_glob_k
    for w_local in w_locals_k:
        value += w_local / len(w_locals_k)
    return value
def bcfreeze(w_locals, net_glob, args):
    '''
    :param w_locals: list of state_dict, 代表各个client的本地模型参数
    :param net_glob: model, 代表global model
    :return:
    '''
    w_glob = net_glob.state_dict()
    for i in range(len(w_locals)):
        for key, value in w_locals[i].items():
            w_locals[i][key] = value - w_glob[key]
    print("num of len w_locals: ", len(w_locals))
    for key in w_glob.keys():
        w_locals_k = [w_local[key].cpu() for w_local in w_locals]
        w_glob_k = w_glob[key].cpu()
        if key in args.bc_layers:
            print("Freeze layer: ", key)
            print("w_locals_k: ", w_locals_k)
            w_glob[key] = flame_layers(w_locals_k, args)
            # w_glob[key] = fedavg_layers(w_locals_k, w_glob_k)

        else:
            print("Average layer: ", key)
            w_glob[key] = fedavg_layers(w_locals_k, w_glob_k)
    return w_glob
