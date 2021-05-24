# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 15:29:54 2021

@author: rashe
"""

import pandas as pd
import numpy as np

def compareMatrices(pixelDic: dict, references: list):
    
    x = pd.DataFrame(np.zeros((len(pixelDic), len(references)*3)))
    
    for i in range(len(pixelDic)):
        for j,ref in enumerate(references):
            x[0 + j*3][i] = np.linalg.norm(pixelDic[ref][:,:,0] - pixelDic[i][:,:,0])
            x[1 + j*3][i] = np.linalg.norm(pixelDic[ref][:,:,1] - pixelDic[i][:,:,1])
            x[2 + j*3][i] = np.linalg.norm(pixelDic[ref][:,:,2] - pixelDic[i][:,:,2])
            
    
    return pd.DataFrame(np.sum(np.array(x),axis=1))