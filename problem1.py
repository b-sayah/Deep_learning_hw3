#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:20:15 2019

@author: chin-weihuang
"""
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad
import matplotlib.pyplot as plt
#from scipy.spatial import distance

import samplers

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),

        )

    def forward(self, x):
        return self.model(x)


class jsd_mlp(nn.Module):
    def __init__(self, input_dim):
        super(jsd_mlp, self).__init__()
        self.model = nn.Sequential(
            MLP(input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def jsd_objective(Discrim, x_p, y_q):

    jsd_objective = torch.log(torch.Tensor([2])) + 0.5 * torch.log(Discrim(x_p)).mean() + 0.5 * torch.log(
        1 - Discrim(y_q)).mean()

    return jsd_objective


def js_divergence(x_p, y_q):
    r = (x_p + y_q) / 2

    D1 = x_p * np.log(x_p / r)
    D2 = y_q * np.log(y_q / r)

    return (np.sum(D1 + D2)) / 2


def Gradient_Penalty(MLP, x_p, y_q):
    # interpolation
    alfa = torch.rand(x_p.shape[0], 1)
    alfa = alfa.expand(-1, MLP.input_dim)

    interpolation = Variable(alfa * x_p + (1 - alfa) * y_q, requires_grad=True)

    inputs = interpolation
    outputs = MLP(interpolation)
    # Gradients calculation
    gradients = grad(outputs, inputs, torch.ones(outputs.size()),
                     retain_graph=True, create_graph=True, only_inputs=True)[0]

    # Mean/Expectation of gradients
    # gradients = gradients.view(gradients.size(0),  -1)
    gradients = gradients.view(gradients.size(0), -1)
    gradients_norm = gradients.norm(2, dim=1)

    # TO DO check it again if we really need this, square root to manually calculate norm+epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients_norm) + 1e-3)

    # mean?
    gradient_penalty = (gradients_norm - 1) ** 2

    return gradient_penalty


def wasserstein_loss(MLP, x_p, y_q, lamda=100):
    y_1 = MLP(x_p)
    y_2 = MLP(y_q)

    GP = Gradient_Penalty(MLP, x_p, y_q)

    return -(y_1.mean() - y_2.mean() - lamda * GP.mean())




