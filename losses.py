# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 14:50:25 2021

@author: rashe
"""

from torch import nn

def loss(net, X_train_rot, X_train_shft, X_train_flip, X_train_jit):

    logits1, logits2 = net.forward(X_train_rot), net.forward(X_train_shft) 
    logits3, logits4 = net.forward(X_train_flip), net.forward(X_train_jit) 
    
    logSoft = nn.LogSoftmax(dim=1)
    soft = nn.Softmax(dim=0)
    
    log_y_giv_x1 = logSoft(logits1)
    log_y_giv_x2 = logSoft(logits2)
    
    log_y_giv_x3 = logSoft(logits3)
    log_y_giv_x4 = logSoft(logits4)
    
    
    x1_giv_y = soft(logits1)
    x2_giv_y = soft(logits2)
    
    x3_giv_y = soft(logits3)
    x4_giv_y = soft(logits4)
    
    def computeCrossEntropy(x1_giv_y, x2_giv_y,log_y_giv_x1,log_y_giv_x2):
        l1 = - (x2_giv_y * log_y_giv_x1).sum() /len(logits1)
        l2 = - (x1_giv_y * log_y_giv_x2).sum() /len(logits2)
        
        return l1, l2
    
    l12, l21 = computeCrossEntropy(x1_giv_y, x2_giv_y, log_y_giv_x1, log_y_giv_x2)
    l13, l31 = computeCrossEntropy(x1_giv_y, x3_giv_y, log_y_giv_x1, log_y_giv_x3)
    l14, l41 = computeCrossEntropy(x1_giv_y, x4_giv_y, log_y_giv_x1, log_y_giv_x4)
    l23, l32 = computeCrossEntropy(x2_giv_y, x3_giv_y, log_y_giv_x2, log_y_giv_x3)
    l24, l42 = computeCrossEntropy(x2_giv_y, x4_giv_y, log_y_giv_x2, log_y_giv_x4)
    l34, l43 = computeCrossEntropy(x3_giv_y, x4_giv_y, log_y_giv_x3, log_y_giv_x4)
    
    
    return (l12 + l21 + l13 + l31 + l14 + l41 + l23 + l32 + l24 + l42 + l34 + l43) / 12

def groundedLoss(net, real, X_train_rot, X_train_shft, X_train_flip, X_train_jit, X_train_col):

    realLogits = net.forward(real)
    
    logits1, logits2 = net.forward(X_train_rot), net.forward(X_train_shft) 
    logits3, logits4 = net.forward(X_train_flip), net.forward(X_train_jit) 
    logits5 = net.forward(X_train_col)
    
    logSoft = nn.LogSoftmax(dim=1)
    soft = nn.Softmax(dim=0)
    
    
    log_y_giv_xreal = logSoft(realLogits)
    log_y_giv_x1 = logSoft(logits1)
    log_y_giv_x2 = logSoft(logits2)
    
    log_y_giv_x3 = logSoft(logits3)
    log_y_giv_x4 = logSoft(logits4)
    
    log_y_giv_x5 = logSoft(logits5)
    
    xreal_giv_y = soft(realLogits)
    x1_giv_y = soft(logits1)
    x2_giv_y = soft(logits2)
    
    x3_giv_y = soft(logits3)
    x4_giv_y = soft(logits4)
    
    x5_giv_y = soft(logits5)
    
    def computeCrossEntropy(x1_giv_y, x2_giv_y,log_y_giv_x1,log_y_giv_x2):
        l1 = - (x2_giv_y * log_y_giv_x1).sum() /len(logits1)
        l2 = - (x1_giv_y * log_y_giv_x2).sum() /len(logits2)
        
        return l1, l2
    
    l1real, lreal1 = computeCrossEntropy(xreal_giv_y, x1_giv_y, log_y_giv_xreal, log_y_giv_x1)
    l2real, lreal2 = computeCrossEntropy(xreal_giv_y, x2_giv_y, log_y_giv_xreal, log_y_giv_x2)
    l3real, lreal3 = computeCrossEntropy(xreal_giv_y, x3_giv_y, log_y_giv_xreal, log_y_giv_x3)
    l4real, lreal4 = computeCrossEntropy(xreal_giv_y, x4_giv_y, log_y_giv_xreal, log_y_giv_x4)
    l5real, lreal5 = computeCrossEntropy(xreal_giv_y, x5_giv_y, log_y_giv_xreal, log_y_giv_x5)
  
    L1 = (l1real + lreal1 + l2real + lreal2 + l3real + lreal3 + l4real + lreal4 + l5real + lreal5) / 10
    
    l12, l21 = computeCrossEntropy(x1_giv_y, x2_giv_y, log_y_giv_x1, log_y_giv_x2)
    l13, l31 = computeCrossEntropy(x1_giv_y, x3_giv_y, log_y_giv_x1, log_y_giv_x3)
    l14, l41 = computeCrossEntropy(x1_giv_y, x4_giv_y, log_y_giv_x1, log_y_giv_x4)
    l23, l32 = computeCrossEntropy(x2_giv_y, x3_giv_y, log_y_giv_x2, log_y_giv_x3)
    l24, l42 = computeCrossEntropy(x2_giv_y, x4_giv_y, log_y_giv_x2, log_y_giv_x4)
    l34, l43 = computeCrossEntropy(x3_giv_y, x4_giv_y, log_y_giv_x3, log_y_giv_x4)
    l15, l51 = computeCrossEntropy(x1_giv_y, x5_giv_y, log_y_giv_x1, log_y_giv_x5)
    l25, l52 = computeCrossEntropy(x2_giv_y, x5_giv_y, log_y_giv_x2, log_y_giv_x5)
    l35, l53 = computeCrossEntropy(x3_giv_y, x5_giv_y, log_y_giv_x3, log_y_giv_x5)
    l45, l54 = computeCrossEntropy(x4_giv_y, x5_giv_y, log_y_giv_x4, log_y_giv_x5)
    
    
    L2 = (l12 + l21 + l13 + l31 + l14 + l41 + l23 + l32 + l24 + l42 + l34 + l43
          + l15 + l51 + l25 + l52 + l35 + l53 + l45 + l54) / 20
    
    return (L1 + L2) / 2