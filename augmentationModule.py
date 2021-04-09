# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 14:58:59 2021

@author: rashe
"""

from skimage.transform import rotate, AffineTransform, warp


def augmentation1(image):
    return rotate(image, angle=15, mode = 'wrap')

def augmentation2(image):
    #return warp(image,transform,mode='wrap')
    return rotate(image, angle=-15, mode = 'wrap')