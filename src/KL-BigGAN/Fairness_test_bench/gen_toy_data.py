# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 09:50:48 2021

@author: Chris

Experiment thoughts:
> 2 Classes Multivariate norm
- with a mean shift but covariance remains the same
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from tqdm import tqdm

directory='./toy_data'

def gen_dist(mean,cov,samples,res,debug):
    x, y = np.random.multivariate_normal(mean, cov, samples).T
    x=np.round(x)
    y=np.round(y)
      
    if debug==1:
        #plt.axis('equal')
        plt.plot(x, y, 'x')
        plt.axis([0,res,0,res])
        plt.show()
    return (x,y)


def make_tensor(mean,cov,samples,res,debug):
    x,y=gen_dist(mean,cov,samples,res,debug)
    matrix=np.zeros((res,res))
    coord=[(x[i],y[i]) for i in range(len(x))]
    for i in coord:
        # print (i[1],i[0])
        matrix[int(i[1])][int(i[0])]=1

    return np.stack((matrix,matrix,matrix))

def make_dataset(mean,cov,samples,res,data_instances,debug):
    datalist=[]
    for i in tqdm(range(data_instances)):
        instance=make_tensor(mean,cov,samples,res,debug)
        datalist.append(instance)
    return np.stack(datalist)
    
    
def create_class_data(num_class,res,samples,data_instances,debug): 

    # device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #Disctively output a center points
    center=int(res/2) #Center spread
    
    #Defined num_class different variances
    min_var=10
    spread_max=center
    # spread_max=min(center,res-center) #Center spread
    var=np.round(np.linspace(min_var,spread_max,num_class))

    if not (os.path.exists(directory)):
        os.makedirs(directory)
    
    class_list=[]
    #Build data classes
    for i in range(num_class):
        mean = [center, center]
        cov = [[var[i], 0], [0, var[i]]]  # diagonal covariance
        dataset=make_dataset(mean,cov,samples,res,data_instances,debug)
        labels=(torch.ones(data_instances)*i).type(torch.uint8)
        dataset=torch.tensor(dataset,device=None).type(torch.uint8)
        class_list.append(dataset)
        torch.save((dataset,labels),os.path.join(directory,("dataset_raw_class_%i")%i))
        
    return class_list

def split_bias_dataset(num_class=2,res=64,samples=100,data_instances=10000,debug=0):
    data_list=create_class_data(num_class,res,samples,data_instances,debug)
    #For all class cofiguration , we will only have one minority dataset
    # for i in np.arange (0.1,1,0.1):
    for i in np.arange (0.1,1,0.05): #class=4
        perc_minority=i
        perc_majority=(1-perc_minority)/(num_class-1)
        
        #Total Cont
        minority_count=round(data_instances*perc_minority)
        majority_count=round(data_instances*perc_majority)
        
        #Random selection index
        sample_index_minority=np.random.choice(data_instances, minority_count)
        sample_index_majority=np.random.choice(data_instances, majority_count)
        
        #Formulate the output List
        bias_list=[]
        label_list=[]
        for j in range(num_class):
            if j==0: #data_minority
                data_minority=np.take(data_list[j],sample_index_minority,axis=0)
                bias_list.append(data_minority)
                label_list.append(torch.zeros(len(data_minority)))
            else:
                data_majority=np.take(data_list[j],sample_index_majority,axis=0)
                bias_list.append(data_majority)
                label_list.append(torch.ones(len(data_majority))*j)
        bias_list=torch.vstack(bias_list)
        label_list=torch.cat(label_list)
        
        torch.save((bias_list,label_list),os.path.join(directory,("test_fairness_data_%f_samples_%i")%(i,data_instances)))
    return (bias_list,label_list)

'''
Sample creation 20 px , 2 classes
'''
def split_bias_dataset_sample(num_class=2,res=64,samples=100,data_instances=20,debug=0,perc_minority=0.5):
    data_list=create_class_data(num_class,res,samples,data_instances,debug)
    #For all class cofiguration , we will only have one minority dataset
    perc_majority=(1-perc_minority)/(num_class-1)
    
    #Total Cont
    minority_count=round(data_instances*perc_minority)
    majority_count=round(data_instances*perc_majority)
    
    #Random selection index
    sample_index_minority=np.random.choice(data_instances, minority_count)
    sample_index_majority=np.random.choice(data_instances, majority_count)
    
    #Formulate the output List
    bias_list=[]
    label_list=[]
    for j in range(num_class):
        if j==0: #data_minority
            data_minority=np.take(data_list[j],sample_index_minority,axis=0)
            bias_list.append(data_minority)
            label_list.append(torch.zeros(len(data_minority)))
        else:
            data_majority=np.take(data_list[j],sample_index_majority,axis=0)
            bias_list.append(data_majority)
            label_list.append(torch.ones(len(data_majority))*j)
    bias_list=torch.vstack(bias_list)
    label_list=torch.cat(label_list)
    
    return (bias_list,label_list)

def reverse_2d_binary_matrix(matrix):
    X=[]
    Y=[]
    for y in range(len(matrix)):
        for x in range(len(matrix[0])):
            if matrix[y][x]!=0:
                X.append(x)
                Y.append(y)
    return (X,Y)

def plot_samples(data_list,num_class=2):
    fig, axs = plt.subplots(num_class, 5,figsize=(15,2.5*num_class))
    res=64
    for i in range (5):
        for j in range(num_class):
            # if (i+5*j==18):
            #     print ("pause")
            #Label=0
            x1,y1=reverse_2d_binary_matrix(data_list[i+5*j][0,:,:])
            axs[j, i].scatter(x1,y1,marker='.')
            axs[j, i].axis([0,res,0,res])
            # #Label=1
            # x2,y2=x1,y2=reverse_2d_binary_matrix(data_list[i+5][0,:,:])
            # axs[1, i].scatter(x2,y2,marker='.')
            # axs[1, i].axis([0,res,0,res])
            
def sample_n_plot(samples=100,class_num=2,perc_minority=0.5):   
    data_instances=5*class_num          
    bias_list,label_list=split_bias_dataset_sample(num_class=class_num,samples=500,data_instances=data_instances,perc_minority=perc_minority)
    plot_samples(bias_list,num_class=class_num)
    
    
# bias_list,label_list=split_bias_dataset(num_class=4,samples=200)
sample_n_plot(samples=100,class_num=4,perc_minority=0.25)