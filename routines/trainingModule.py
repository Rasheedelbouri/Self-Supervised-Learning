# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 14:55:29 2021

@author: rashe
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

from data.curriculumModule import compareMatrices
from data.augmentationModule import augmentation1, augmentation2, augmentation3, augmentation4, augmentation5, augmentation6
from data.formattingModule import formatTorch
from model.losses import expandedLoss, groundedLoss, originalLoss

def train(net,optimizer,pixelDic,curric=False,batchsize=32,numEpochs=10,curricRepeats=1, relativeLoss=False):

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
            flips = []
            jits = []
            scas = []
            cols = []
            if relativeLoss:
                relLossBatch = []
            for i in sa:
                im = pixelDic[i]
                if relativeLoss:
                    relLossBatch.append(formatTorch(im))
                rot,shft, flip, jit = augmentation1(im), augmentation2(im), augmentation3(im), augmentation4(im)
                #sca = augmentation5(im),
                col =  augmentation6(im)
                
                rots.append(formatTorch(rot))
                shfts.append(formatTorch(shft))
                flips.append(formatTorch(flip))
                jits.append(formatTorch(jit))
                #scas.append(formatTorch(sca))
                cols.append(formatTorch(col))
            
            X_train_rot = torch.stack([torch.from_numpy(np.array(i)) for i in rots])
            X_train_shft = torch.stack([torch.from_numpy(np.array(i)) for i in shfts])
            X_train_flip = torch.stack([torch.from_numpy(np.array(i)) for i in flips])
            X_train_jit = torch.stack([torch.from_numpy(np.array(i)) for i in jits])
            X_train_col = torch.stack([torch.from_numpy(np.array(i)) for i in cols])

            if relativeLoss:
                realBatch = torch.stack([torch.from_numpy(np.array(i)) for i in relLossBatch])

            if curric:
                for z in range(curricRepeats):
                
                    optimizer.zero_grad()
                    if relLossBatch:
                        L = originalLoss(net, realBatch, X_train_rot, X_train_shft, X_train_flip, X_train_jit, X_train_col)
                    else:
                        L = originalLoss(net, X_train_rot, X_train_shft, X_train_flip, X_train_jit, X_train_col)
            else:
                optimizer.zero_grad()
                if relLossBatch:
                    L = groundedLoss(net, realBatch, X_train_rot, X_train_shft, X_train_flip, X_train_jit, X_train_col)
                else:
                    L = groundedLoss(net, X_train_rot, X_train_shft, X_train_flip, X_train_jit, X_train_col)
# =============================================================================

# =============================================================================
            L.backward()
            optimizer.step()
    
    return net