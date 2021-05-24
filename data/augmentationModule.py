# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 14:58:59 2021

@author: rashe
"""

from skimage.transform import rotate, rescale
from skimage.color import convert_colorspace
import numpy as np
import cv2


def augmentation1(image):
    return rotate(image, angle=5, mode = 'wrap')

def augmentation2(image):
    noise = np.random.randint(0,50,(224, 224)) # design jitter/noise here
    zitter = np.zeros_like(image)
    zitter[:,:,1] = noise
    noise_added = cv2.add(image, zitter)
    
    return noise_added

def augmentation3(image):
    return np.flip(image, axis=1)

def augmentation4(image):
    return rotate(image, angle=-15, mode = 'wrap')

def augmentation5(image):
    return rescale(image, 0.5, anti_aliasing=False)

def augmentation6(image):
    return convert_colorspace(image, 'RGB', 'HSV')
