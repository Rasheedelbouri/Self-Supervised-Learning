# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 14:55:29 2021

@author: rashe
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch import nn

from curriculumModule import compareMatrices
from augmentationModule import augmentation1, augmentation2
from formattingModule import formatTorch

def train(net,optimizer,pixelDic,curric=False,batchsize=32,numEpochs=10,curricRepeats=1):

    for epoch in tqdm(range(numEpochs)):
        if curric:
            rankings = compareMatrices(pixelDic, [163, 12, 21, 35, 53])
            rankings = rankings.sort_values(0)
                 
        sampleAxis = pd.DataFrame(np.arange(len(pixelDic)))
        
        if batchsize > len(sampleAxis):
            batchsize = len(sampleAxis)
        
        while len(sampleAxis) > 0:
            
            
            try:
                if curric:
                    if len(sampleAxis) < batchsize:
                        sa = list(np.array(rankings[0:len(sampleAxis)].index))
                    else:
                        sa = list(np.array(rankings[0:batchsize].index))
                else:
                    sa = list(np.random.choice(sampleAxis[0],batchsize, replace=False))
            except:
                sa = list(np.random.choice(sampleAxis[0],len(sampleAxis), replace=False))
            
            if curric:
                rankings = rankings.drop(sa)
            sampleAxis = sampleAxis[~sampleAxis[0].isin(sa)]
            
            rots = []
            shfts = []
            for i in sa:
                im = pixelDic[i]
                rot,shft = augmentation1(im), augmentation2(im)
                rot, shft = formatTorch(rot), formatTorch(shft)
                rots.append(rot)
                shfts.append(shft)
            
            X_train_rot = torch.stack([torch.from_numpy(np.array(i)) for i in rots])
            X_train_shft = torch.stack([torch.from_numpy(np.array(i)) for i in shfts])
            
            
            for z in range(curricRepeats):
            
                optimizer.zero_grad()
                
                logits1, logits2 = net.forward(X_train_rot), net.forward(X_train_shft) 
                
                logSoft = nn.LogSoftmax(dim=1)
                soft = nn.Softmax(dim=0)
                
                log_y_giv_x1 = logSoft(logits1)
                log_y_giv_x2 = logSoft(logits2)
                
                x1_giv_y = soft(logits1)
                x2_giv_y = soft(logits2)
                
                l1 = - (x2_giv_y * log_y_giv_x1).sum() /len(logits1)
                l2 = - (x1_giv_y * log_y_giv_x2).sum() /len(logits2)
                L = (l1 + l2) / 2
                L.backward()
                optimizer.step()
    return net