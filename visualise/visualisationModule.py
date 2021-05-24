# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 15:28:49 2021

@author: rashe
"""

import matplotlib.pyplot as plt

def visualiseImage(imDic, imDicPosition):
    plt.imshow(imDic[imDicPosition])
    plt.axis('off')
    plt.show()
