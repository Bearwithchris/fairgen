# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 16:11:07 2021

@author: Chris
"""

import torch
import numpy as np
import time
import os
import matplotlib.pyplot as plt

def load_data(perc_f=0.5,sample_size=10000):
    data=torch.load('./real_data/test_fairness_data_%f_%f_samples_%i'%(round(perc_f,2),round(1-perc_f,2),sample_size))
    print ("Data loaded: "+'./real_data/test_fairness_data_%f_%f_samples_%i'%(round(perc_f,2),round(1-perc_f,2),sample_size))
    dataset=data[0]
    train_set = torch.utils.data.TensorDataset(dataset)
    return (train_set,len(data),data[1])

def load_data_toy(perc_f=0.5,sample_size=10000):
    directory='./toy_data'
    path=os.path.join(directory,("test_fairness_data_%f_samples_%i")%(perc_f,sample_size))
    data=torch.load(path)
    print ("Data loaded: "+ path)
    dataset=data[0]
    train_set = torch.utils.data.TensorDataset(dataset)
    return (train_set,len(data),data[1])

def load_data_toy_ref(perc_f=0.5,sample_size=10000):
    directory='./toy_data'
    path=os.path.join(directory,("test_fairness_ref_data_%f_samples_%i")%(perc_f,sample_size))
    data=torch.load(path)
    print ("Data loaded: "+ path)
    dataset=data[0]
    train_set = torch.utils.data.TensorDataset(dataset)
    return (train_set,len(data),data[1])
# data,__,labels=load_data_toy(0.9)
# print(len(np.where(labels==1)[0]))
# print(len(np.where(labels==0)[0]))
    