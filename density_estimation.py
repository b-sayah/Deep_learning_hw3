#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
from torch.autograd import grad
import numpy as np
import sys
from samplers import *

import matplotlib.pyplot as plt

#Utils function

def jsd_objective(Discrim, x_p, y_q):
    jsd_objective = torch.log(torch.Tensor([2])) + 0.5 * torch.log(Discrim(x_p)).mean() + 0.5 * torch.log(
        1 - Discrim(y_q)).mean()

    return jsd_objective


def wd_objective(Critic, x_p, y_q):
    wd_objective = Critic(x_p).mean() - Critic(y_q).mean()

    return wd_objective


def gradient_penalty(Critic, x_p, y_q, lamda):
    alfa = x_p.size()[0]
    alfa = torch.rand(alfa, 1, device=x_p.device)
    alfa = alfa.expand_as(x_p)

    interpolate_z = Variable(alfa * x_p + (1 - alfa) * y_q, requires_grad=True)

    inputs = interpolate_z
    outputs = Critic(interpolate_z)

    gradients = grad(outputs, inputs, torch.ones(Critic(interpolate_z).size()),
                     create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_norm = gradients.norm(2, dim=1)

    GP = lamda * ((gradient_norm - 1) ** 2).mean()
    return GP


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.model(x)


class jsd_mlp(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(jsd_mlp, self).__init__()
        self.model = nn.Sequential(
            MLP(input_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

#Q1.1
def js_divergence(p, q, m_minibatch=1000):
    x_p = next(p)
    y_q = next(q)
    x_p = torch.Tensor(x_p)
    y_q = torch.Tensor(y_q)

    Discrim = jsd_mlp(input_dim=x_p.size()[1], hidden_dim=32)
    # optimizer_D = torch.optim.Adagrad(D.parameters())
    # optimizer_D = torch.optim.Adam(D.parameters())
    optimizer_D = torch.optim.Adagrad(Discrim.parameters())

    for mini_batch in range(m_minibatch):
        optimizer_D.zero_grad()
        jsd_loss = jsd_objective(Discrim, x_p, y_q)

        jsd_loss.backward(torch.FloatTensor([-1]))
        optimizer_D.step()

    #jsd = jsd_objective(Discrim, x_p, y_q)
    return Discrim, jsd_objective(Discrim, x_p, y_q)

#Q1.2
def w_distance(p, q, m_minibatch=1000, lamda=10):
    x_p = next(p)
    y_q = next(q)
    x_p = torch.Tensor(x_p)
    y_q = torch.Tensor(y_q)

    Critic = MLP(input_dim=x_p.size()[1], hidden_dim=64)
    # optimizer_T = torch.optim.SGD(T.parameters(), lr=1e-3)
    # optimizer_T = torch.optim.Adam(T.parameters())
    optimizer_T = torch.optim.Adagrad(Critic.parameters())

    for mini_batch in range(m_minibatch):
        optimizer_T.zero_grad()
        wd = wd_objective(Critic, x_p, y_q)
        wd_loss = wd - gradient_penalty(Critic, x_p, y_q, lamda)

        wd_loss.backward(torch.FloatTensor([-1]))
        optimizer_T.step()

    #wd = wd_objective(Critic, x_p, y_q)
    #GP = gradient_penalty(Critic, x_p, y_q, lamda)
    #wd = wd - GP
    return Critic, wd_objective(Critic, x_p, y_q) - gradient_penalty(Critic, x_p, y_q, lamda)


# Q1.3

Phi_values = [-1 + 0.1 * i for i in range(21)]

estimated_jsd, estimated_wd = [], []

m_minibatch = 1000
batch_size = 512
lamda = 10


for Phi in Phi_values:
    # TO DO
    dist_p = distribution1(0, batch_size)

    dist_q = distribution1(Phi, batch_size)

    Discrim, jsd = js_divergence(dist_p, dist_q, m_minibatch)
    estimated_jsd.append(jsd)

    Critic, wd = w_distance(dist_p, dist_q, m_minibatch, lamda)
    estimated_wd.append(wd)

    print(f"Phi: {Phi:.2f}  estimated JSD: {jsd.item():.6f}  estimated WD: {wd.item():.6f}")  # TO DO

plt.figure(figsize=(8, 4))
plt.plot(Phi_values, estimated_jsd)
plt.plot(Phi_values, estimated_wd)
plt.title('JSD and WD in terms of phi')
plt.xlabel('Phi values')
plt.ylabel('estimate')
plt.legend(["estimated JSD", "estimated WD"])

plt.savefig('estimated JSD & WD.png')
plt.show()


# Q1.4
# plot p0 and p1
plt.figure()

# empirical
xx = torch.randn(10000)
f = lambda x: torch.tanh(x*2+1) + x*0.75
d = lambda x: (1-torch.tanh(x*2+1)**2)*2+0.75
plt.hist(f(xx), 100, alpha=0.5, density=1)
plt.hist(xx, 100, alpha=0.5, density=1)
plt.xlim(-5,5)
# exact
xx = np.linspace(-5,5,1000)
N = lambda x: np.exp(-x**2/2.)/((2*np.pi)**0.5)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.plot(xx, N(xx))


############### import the sampler ``samplers.distribution4'' 
############### train a discriminator on distribution4 and standard gaussian
############### estimate the density of distribution4

#######--- INSERT YOUR CODE BELOW ---#######
 

batch_size = 512# 10240
m_minibatch = 100 #1000
f0 = iter(distribution3(batch_size))
f1 = iter(distribution4(batch_size))

Discrim, jsd = js_divergence(f1, f0, m_minibatch)


############### plotting things
############### (1) plot the output of your trained discriminator 
############### (2) plot the estimated density contrasted with the true density

# evaluate xx using your discriminator; replace xx with the outpu
r = Discrim(torch.Tensor(xx).unsqueeze(dim=1)).detach().numpy().reshape(-1)
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(xx,r)
plt.title(r'$D(x)$')

estimate = estimate = N(xx) * r/(1-r)# estimate the density of distribution4 (on xx) using the discriminator;
                                # replace "np.ones_like(xx)*0." with your estimate
plt.subplot(1,2,2)
plt.plot(xx,estimate)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.legend(['Estimated','True'])
plt.title('Estimated vs True')











