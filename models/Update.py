#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from models.add_trigger import add_trigger
import numpy as np
import random
from sklearn import metrics
import copy
import math


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.poison_frac = args.poison_frac
        self.attack_label = args.attack_label
        self.attack_goal = args.attack_goal
        self.attack_method = args.attack_method

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def add_trigger(self, image):
        return add_trigger(self.args, image)
    def trigger_data(self, images, labels):
        #  attack_goal == -1 means attack all label to attack_label
        if self.attack_goal == -1:
            if math.isclose(self.poison_frac, 1):  # 100% copy poison data
                bad_data, bad_label = copy.deepcopy(
                        images), copy.deepcopy(labels)
                for xx in range(len(bad_data)):
                    bad_label[xx] = self.attack_label
                    # bad_data[xx][:, 0:5, 0:5] = torch.max(images[xx])
                    bad_data[xx] = self.add_trigger(bad_data[xx])
                images = torch.cat((images, bad_data), dim=0)
                labels = torch.cat((labels, bad_label))
            else:
                for xx in range(len(images)):  # poison_frac% poison data
                    labels[xx] = self.attack_label
                    # images[xx][:, 0:5, 0:5] = torch.max(images[xx])
                    images[xx] = self.add_trigger(images[xx])
                    if xx > len(images) * self.poison_frac:
                        break
        else:  # trigger attack_goal to attack_label
            if math.isclose(self.poison_frac, 1):  # 100% copy poison data
                bad_data, bad_label = copy.deepcopy(
                        images), copy.deepcopy(labels)
                for xx in range(len(bad_data)):
                    if bad_label[xx]!= self.attack_goal:  # no in task
                        continue  # jump
                    bad_label[xx] = self.attack_label
                    bad_data[xx] = self.add_trigger(bad_data[xx])
                    images = torch.cat((images, bad_data[xx].unsqueeze(0)), dim=0)
                    labels = torch.cat((labels, bad_label[xx].unsqueeze(0)))
            else:  # poison_frac% poison data
                # count label == goal label
                num_goal_label = len(labels[labels==self.attack_goal])
                counter = 0
                for xx in range(len(images)):
                    if labels[xx] != 0:
                        continue
                    labels[xx] = self.attack_label
                    # images[xx][:, 0:5, 0:5] = torch.max(images[xx])
                    images[xx] = self.add_trigger(images[xx])
                    counter += 1
                    if counter > num_goal_label * self.poison_frac:
                        break
        return images, labels
    def train_attack(self, net, args):
        badnet = copy.deepcopy(net)
        badnet.train()
        good_param = copy.deepcopy(net.state_dict())
        optimizer = torch.optim.SGD(badnet.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                bad_data, bad_label = copy.deepcopy(
                    images), copy.deepcopy(labels)
                images, labels = self.trigger_data(images, labels)
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                badnet.zero_grad()
                log_probs = badnet(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        bad_net_param = badnet.state_dict()
        self.malicious_model = copy.deepcopy(badnet)

        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        self.benign_model = copy.deepcopy(net)
        attack_param = {}
        attack_list = args.bc_layers
        for key, value in net.state_dict().items():
            if key in attack_list:
                attack_param[key] = bad_net_param[key]
            else:
                attack_param[key] = good_param[key]
        return attack_param, sum(epoch_loss) / len(epoch_loss)
