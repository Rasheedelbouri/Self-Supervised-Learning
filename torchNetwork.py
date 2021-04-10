# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 18:02:59 2021

@author: rashe
"""
import torch
from torch.optim import Adam
from loadData import getDataSet
from visualisationModule import visualiseImage
from trainingModule import train
from testModule import test
from architectureModule import customNet
from tensorflow.keras.datasets import cifar10


foldername = 'cifar' #adding a comment

if foldername.lower() == 'cifar':
    (_,_), (pixelDic,_) = cifar10.load_data()
else:
    imDic, pixelDic = getDataSet(foldername)

batchsize = 64
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

j=2
for i in range(len(clusters[j])):
    visualiseImage(imDic, clusters[j][i])