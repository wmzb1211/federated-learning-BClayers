#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from models.add_trigger import add_trigger
from skimage import img_as_ubyte
from skimage import io


# def test_img(net_g, datatest, args):
#     net_g.eval()
#     # testing
#     test_loss = 0
#     correct = 0
#     data_loader = DataLoader(datatest, batch_size=args.bs)
#     l = len(data_loader)
#     for idx, (data, target) in enumerate(data_loader):
#         if args.gpu != -1:
#             data, target = data.cuda(), target.cuda()
#         log_probs = net_g(data)
#         # sum up batch loss
#         test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
#         # get the index of the max log-probability
#         y_pred = log_probs.data.max(1, keepdim=True)[1]
#         correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
#
#     test_loss /= len(data_loader.dataset)
#     accuracy = 100.00 * correct / len(data_loader.dataset)
#     if args.verbose:
#         print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
#             test_loss, correct, len(data_loader.dataset), accuracy))
#     return accuracy, test_loss
def save_img(image):
    img = image
    if image.shape[0] == 1:
        pixel_min = torch.min(img)
        img -= pixel_min
        pixel_max = torch.max(img)
        img /= pixel_max
        io.imsave('./save/test_trigger2.png', img_as_ubyte(img.squeeze().cpu().numpy()))
    else:
        img = image.cpu().numpy()
        img = img.transpose(1, 2, 0)
        pixel_min = np.min(img)
        img -= pixel_min
        pixel_max = np.max(img)
        img /= pixel_max
        io.imsave('./save/test_trigger2.png', img_as_ubyte(img))

def test_or_not(args, label):
    if args.attack_goal != -1:  # one to one
        if label == args.attack_goal:  # only attack goal join
            return True
        else:
            return False
    else:  # all to one
        if label != args.attack_label:
            return True
        else:
            return False
def test_img(net_g, datatest, args, test_backdoor=False):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    back_correct = 0
    back_num = 0
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        if test_backdoor:
            for k, image in enumerate(data):
                if test_or_not(args, target[k]):  # one2one need test
                    data[k] = add_trigger(args,data[k], test=True)
                    save_img(data[k])
                    target[k] = args.attack_label
                    back_num += 1
                else:
                    target[k] = -1
            log_probs = net_g(data)
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            if args.defence == 'flip':
                soft_max_probs = torch.nn.functional.softmax(log_probs.data, dim=1)
                pred_confidence = torch.max(soft_max_probs, dim=1)
                x = torch.where(pred_confidence.values > 0.4,pred_confidence.indices, -2)
                back_correct += x.eq(target.data.view_as(x)).long().cpu().sum()
            else:
                back_correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    if test_backdoor:
        back_accu = 100.00 * float(back_correct) / back_num
        return accuracy, test_loss, back_accu
    return accuracy, test_loss