# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 15:04:56 2021

@author: rashe
"""

import numpy as np
import pandas as pd
from formattingModule import testFormatTorch

def test(net, pixelDic):

    preds =[]
    for i in range(len(pixelDic)):
        ten = net.forward(testFormatTorch(pixelDic[i]))
        preds.append(np.argmax(ten.detach().numpy()[0]))
    
    
    preds = pd.DataFrame(preds)
    uniques = pd.DataFrame(preds[0].unique())
    uniques['count'] = uniques[0].map(preds[0].value_counts())
    
    indices = {}
    for l in range(len(uniques)):
        indices[l] = preds[preds[0] == uniques[0][l]].index
        
    return indices