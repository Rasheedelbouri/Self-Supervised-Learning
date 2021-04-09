# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 18:02:59 2021

@author: rashe
"""

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms

from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

from loadData import getDataSet
from visualisationModule import visualiseImage
from skimage.transform import rotate, AffineTransform, warp
import matplotlib.pyplot as plt
from trainingModule import train
from testModule import test
from architectureModule import customNet

import numpy as np
import pandas as pd

transform = AffineTransform(translation=(15,15))


imDic, pixelDic = getDataSet('Cyrene')

batchsize = 20
numEpochs=100
curric = False
curricRepeats = 1
if not curric:
    curricRepeats = 1

net = customNet()

optimizer = Adam(net.parameters(), lr=0.001)

net = train(net,optimizer,pixelDic,curric,batchsize,numEpochs,curricRepeats)

torch.save(net.state_dict(), "savedTorchModel")

clusters = test(net,pixelDic)