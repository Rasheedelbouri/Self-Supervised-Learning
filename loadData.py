# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 18:54:57 2021

@author: rashe
"""

import pandas as pd
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt


class dataLoader():
    
    def __init__(self, foldername):

        self.path = os.path.join('../../../../Downloads/', str(foldername)+'/'+str(foldername))
        self.imList = pd.DataFrame(os.listdir(self.path))
        self.imList = list(self.imList[self.imList[0].str.contains(".jp")][0])
    
    
    def openImage(self, imFile):
        im = Image.open(os.path.join(self.path, imFile))
        im = im.resize([224,224])
        
        return im
    
    def gatherImages(self):
        imDic = {}
        pixelDic = {}
        for i, im in enumerate(self.imList):
            imDic[i] = self.openImage(im)
            pixelDic[i] = np.array(imDic[i].getdata()).reshape(imDic[i].size[0], imDic[i].size[1], 3)
            
        return imDic, pixelDic
    


def getDataSet(foldername):
    dl = dataLoader(foldername)
    imDic, pixelDic = dl.gatherImages()

    return imDic, pixelDic

        
    
    