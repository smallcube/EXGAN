import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import torch.nn.functional as F

def loss_dis_real(dis_real, out_cat, y, weights=None, cat_weight=None, gamma=2.0):
    #step 1: the loss for GAN
    logpt = F.softplus(-dis_real)
    pt = torch.exp(-logpt)
    if weights is None:
        weights = pt.detach()
        p = pt*1
        #print(dis_real.shape)
        #print(p.shape)
        p = p.view(dis_real.shape[0], 1)
        p = (1-p)**gamma
        loss = p.clone().detach() * logpt
    else:
        weights = torch.cat([weights, pt], 1)
        p = torch.mean(weights, 1)
        p = p.view(len(dis_real), 1)
        p = (1-p)**gamma
        loss = p.clone().detach()*logpt
    loss = torch.mean(loss)

    #step 2: loss for classifying
    target = y.view(y.size(0), 1)
    pt_cat = (1.-target.float())*(1-out_cat) + target.float()*out_cat
    logpt_cat = -torch.log(pt_cat)
    batch_size = target.size(0)
    
    if cat_weight is None:
        cat_weight = pt_cat.detach()
        p = pt_cat*1
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
    else:
        cat_weight = torch.cat([cat_weight, pt_cat], 1)
        p = torch.mean(cat_weight, 1)
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
    logpt_cat = p.clone().detach()*logpt_cat
    loss_cat = torch.mean(logpt_cat)
    return loss, loss_cat, weights, cat_weight

def loss_dis_real_softmax(dis_real, out_cat, y, weights=None, cat_weight=None, gamma=2.0):
    #step 1: the loss for GAN
    batch_size = dis_real.shape[0]
    logpt = F.softplus(-dis_real)
    pt = torch.exp(-logpt)
    if weights is None:
        weights = pt.detach()
        p = pt*1
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
        loss = p * logpt
    else:
        weights = torch.cat([weights, pt], 1)
        p = torch.mean(weights, 1)
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
        loss = p*logpt
    loss = torch.mean(loss)

    #step 2: loss for classifying
    logpt_cat  = F.log_softmax(out_cat, dim=1)
    pt_cat = torch.exp(logpt_cat)
    index = y.view(batch_size, 1).long()
    pt_cat = pt_cat.gather(1, index)
    
    if cat_weight is None:
        cat_weight = pt_cat.detach()
        modulating_factor = pt_cat*1
    else:
        cat_weight = torch.cat([cat_weight, pt_cat], 1)
        modulating_factor = torch.mean(cat_weight, 1)
    modulating_factor = modulating_factor.view(batch_size, 1)
    modulating_factor = (1-modulating_factor)**gamma
    logpt_cat = modulating_factor*logpt_cat

    loss_cat = F.nll_loss(logpt_cat, y)
    
    return loss, loss_cat, weights, cat_weight

def loss_dis_fake(dis_real, out_cat, y, weights=None, cat_weight=None, gamma=2.0):
    #step 1: the loss for GAN
    logpt = F.softplus(dis_real)
    pt = torch.exp(-logpt)
    if weights is None:
        weights = pt.detach()
        p = pt*1
        #print(dis_real.shape)
        #print(p.shape)
        p = p.view(dis_real.shape[0], 1)
        p = (1-p)**gamma
        loss = p.clone().detach() * logpt
    else:
        weights = torch.cat([weights, pt], 1)
        p = torch.mean(weights, 1)
        p = p.view(len(dis_real), 1)
        p = (1-p)**gamma
        loss = p.clone().detach()*logpt
    loss = torch.mean(loss)

    #step 2: loss for classifying
    target = y.view(y.size(0), 1)
    pt_cat = (1.-target.float())*(1-out_cat) + target.float()*out_cat
    logpt_cat = -torch.log(pt_cat)
    batch_size = target.size(0)
    
    if cat_weight is None:
        cat_weight = pt_cat.detach()
        p = pt_cat*1
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
    else:
        cat_weight = torch.cat([cat_weight, pt_cat], 1)
        p = torch.mean(cat_weight, 1)
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
    logpt_cat = p.clone().detach()*logpt_cat
    loss_cat = torch.mean(logpt_cat)
    return loss, loss_cat, weights, cat_weight

def loss_dis_fake_softmax(dis_real, out_cat, y, weights=None, cat_weight=None, gamma=2.0):
    #step 1: the loss for GAN
    batch_size = dis_real.shape[0]
    logpt = F.softplus(dis_real)
    pt = torch.exp(-logpt)
    if weights is None:
        weights = pt.detach()
        p = pt*1
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
        loss = p * logpt
    else:
        weights = torch.cat([weights, pt], 1)
        p = torch.mean(weights, 1)
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
        loss = p*logpt
    loss = torch.mean(loss)

    #step 2: loss for classifying
    logpt_cat  = F.log_softmax(out_cat, dim=1)
    pt_cat = torch.exp(logpt_cat)
    index = y.view(batch_size, 1).long()
    pt_cat = pt_cat.gather(1, index)
    
    if cat_weight is None:
        cat_weight = pt_cat.detach()
        modulating_factor = pt_cat*1
    else:
        cat_weight = torch.cat([cat_weight, pt_cat], 1)
        modulating_factor = torch.mean(cat_weight, 1)
    modulating_factor = modulating_factor.view(batch_size, 1)
    modulating_factor = (1-modulating_factor)**gamma
    logpt_cat = modulating_factor*logpt_cat

    loss_cat = F.nll_loss(logpt_cat, y)
    
    return loss, loss_cat, weights, cat_weight