# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 18:02:59 2021

@author: rashe
"""
import torch
from torch.optim import Adam, SGD
from loadData import getDataSet
from trainingModule import train
from testModule import test
from architectureModule import customNet, loadedPreTrained
from tensorflow.keras.datasets import cifar10
from visualisationModule import visualiseImage


foldername = 'Shapes' #adding a comment

if foldername.lower() == 'cifar':
    (_,_), (pixelDic,_) = cifar10.load_data()
    pixelDic = pixelDic/255
else:
    imDic, pixelDic = getDataSet(foldername)

batchsize = 10
numEpochs=100
curric = False
curricRepeats = 10
relativeLoss = True
loadModel = True
save = False
finetune = True
if not curric:
    curricRepeats = 1

if loadModel:
    net = loadedPreTrained('resnet', finetune)
else:
    net = customNet()

if finetune:
    optimizer = SGD(net.model_ft.fc.parameters(), lr=0.1)
else:
    optimizer = SGD(net.model_ft.parameters(), lr=0.1)


net = train(net,optimizer,pixelDic,curric,batchsize,numEpochs,curricRepeats,relativeLoss)
if save:
    torch.save(net.state_dict(), "savedTorchModel")

clusters = test(net,pixelDic)

