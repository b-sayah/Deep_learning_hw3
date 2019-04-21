#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad

from samplers import *

import matplotlib.pyplot as plt



def jsd_objective(Discrim, x_p, y_q):

    jsd_objective = torch.log(torch.Tensor([2])) + 0.5 * torch.log(Discrim(x_p)).mean() + 0.5 * torch.log(
        1 - Discrim(y_q)).mean()

    return jsd_objective

def wd_objective(Critic, x_p, y_q):

    wd_objective = Critic(x_p).mean() - Critic(y_q).mean()

    return wd_objective

def gradient_penalty(MLP, x_p, y_q):

    alfa = x_p.size()[0]
    alfa = torch.rand(alfa, 1)
    alfa.expand_as(x_p)

    interpolate_z = Variable(alfa * x_p + (1 - alfa) * y_q, requires_grad=True)

    inputs = interpolate_z
    outputs = Critic(interpolate_z)

    gradients = grad(outputs, inputs, torch.ones(Critic(interpolate_z).size()),
                     create_graph=True, retain_graph=True, only_inputs=True)[0]

    # gradients = gradients.view(gradients.size(0),  -1)
    gradient_norm = gradients.norm(2, dim=1)
    GP = lamda * ((gradient_norm - 1) ** 2).mean()
    return GP

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),

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

class wd_mlp(nn.Module):
    def __init__(self, input_dim):
        super(wd_mlp, self).__init__()
        self.model = nn.Sequential(
            MLP(input_dim),
            F.ReLU()
        )

    def forward(self, x):
        return self.model(x)

def js_divergence(p, q, m_minibatch=1000):

    x_p = next(p)
    y_q = next(q)
    x_p = torch.Tensor(x_p)
    y_q = torch.Tensor(y_q)

    Discrim = jsd_mlp(input_dim=x_p.size()[1])
    optimizer_D = torch.optim.Adagrad(Discrim.parameters())
    Discrim.zero_grad()

    for mini_batch in range(m_minibatch):
        jsd_loss = jsd_objective(Discrim, x_p, y_q)

        jsd_loss.backward(torch.FloatTensor([-1]))
        optimizer_D.step()

    jsd_exp = jsd_objective(Discrim, x_p, y_q)
    return Discrim, jsd_exp

def w_distance(p, q, m_minibatch=1000, lamda=10):
    x_p = next(p)
    y_q = next(q)
    x_p = torch.Tensor(x_p)
    y_q = torch.Tensor(y_q)

    Critic = MLP(input_dim=x_p.size()[1])
    optimizer_T = torch.optim.SGD(T.parameters(), lr=1e-3)
    Critic.zero_grad()
    for mini_batch in range(m_minibatch):

        wd = wd_objective(Critic, x_p, y_q)
        wd_loss = wd - gradient_penalty(Critic, x_p, y_q, lamda)

        wd_loss.backward(torch.FloatTensor([-1]))
        optimizer_T.step()

    wd = wd_objective(Critic, x_p, y_q)
    GP = gradient_penalty(Critic, x_p, y_q, lamda)

    return Critic, wd - GP

"""
def q1_3():
    metrics = {"JS": [], "WD": []}
    thetas = [-1 + 0.1 * i for i in range(21)]
    for theta in thetas:
        p = samplers.distribution1(0)
        q = samplers.distribution1(theta)
        D = MLP()
        for metric, values in metrics.items():
            values.append(- D.fit(p, q, metric))  # TODO IMPLEMENT

    for metric, values in metrics.items():
        plt.figure(figsize=(8, 4))
        plt.subplot()
        plt.plot(thetas, values)
        plt.savefig(metric + ".png")
"""
#q1.3
