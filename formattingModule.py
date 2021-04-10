# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 14:59:50 2021

@author: rashe
"""

import torch
import numpy as np

def formatTorch(image):
    z = image.reshape(3,32,32)
    z = torch.tensor(z)
    z = z.float()
    return z

def testFormatTorch(image):
    z = image.reshape(3,32,32)
    z = np.expand_dims(z, axis=0)
    z = torch.tensor(z)
    z = z.float()
    return z