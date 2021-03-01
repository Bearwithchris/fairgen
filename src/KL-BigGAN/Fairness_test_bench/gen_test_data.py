# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:27:51 2021

@author: Chris
"""
import torch 
import numpy as np
import os

BASE_PATH = '../../../data/'
def test_gen_ref(perc_f=0.5,sample_size=10000):
    """
    Returns a dataset used for classification for given class label <class_idx>. If class_idx2 is not None, returns both labels (this is typically used for downstream tasks)
    
    Args:
        split (str): one of [train, val, test]
        class_idx (int): class label for protected attribute
        class_idx2 (None, optional): additional class for downstream tasks
    
    Returns:
        TensorDataset for training attribute classifier
    """
    class_idx=20
    split="test"
    data = torch.load(BASE_PATH + '{}_celeba_64x64.pt'.format(split))
    labels = torch.load(BASE_PATH + '{}_labels_celeba_64x64.pt'.format(split))
    labels1 = labels[:, class_idx]
    
    #Seiving out the male_female breakdown
    M_F_labels=labels.numpy()[:,20]
    male=np.where(M_F_labels==1)
    female=np.where(M_F_labels==0)
    
    #Split counts
    #total_Count=len(male[0])+len(female[0])
    m_Count=round(sample_size*(1-perc_f))
    f_Count=round(sample_size*perc_f)
    
    #Extracting data
    data_M=np.take(data,male,axis=0)[0]
    data_F=np.take(data,female,axis=0)[0]
    
    #Random selection index
    sample_index_M=np.random.choice(len(data_M), int(m_Count))
    sample_index_F=np.random.choice(len(data_F), int(f_Count))
    data_M=np.take(data_M,sample_index_M,axis=0)
    data_F=np.take(data_F,sample_index_F,axis=0)
    rebalanced_dataset=torch.cat((data_M,data_F),axis=0)
    
    #Newlabels
    label_M=torch.ones(len(data_M))
    label_F=torch.zeros(len(data_F))
    rebalanced_labels=torch.cat((label_M,label_F))
    # torch.save((rebalanced_dataset,rebalanced_labels),'./test_fairness_data_%f_%f_samples_%i'%(round(perc_f,2),round(1-perc_f,2),sample_size))

    if not (os.path.exists('./real_data')):
        os.makedirs('./real_data')
        
    torch.save((rebalanced_dataset,rebalanced_labels),'./real_data/test_fairness_ref_data_%f_%f_samples_%i'%(round(perc_f,2),round(1-perc_f,2),sample_size))

    # return appropriate split

    dataset = torch.utils.data.TensorDataset(rebalanced_dataset, rebalanced_labels)
    return dataset

def test_gen(perc_f=0.5,sample_size=10000):
    """
    Returns a dataset used for classification for given class label <class_idx>. If class_idx2 is not None, returns both labels (this is typically used for downstream tasks)
    
    Args:
        split (str): one of [train, val, test]
        class_idx (int): class label for protected attribute
        class_idx2 (None, optional): additional class for downstream tasks
    
    Returns:
        TensorDataset for training attribute classifier
    """
    class_idx=20
    split="test"
    data = torch.load(BASE_PATH + '{}_celeba_64x64.pt'.format(split))
    labels = torch.load(BASE_PATH + '{}_labels_celeba_64x64.pt'.format(split))
    labels1 = labels[:, class_idx]
    
    #Seiving out the male_female breakdown
    M_F_labels=labels.numpy()[:,20]
    male=np.where(M_F_labels==1)
    female=np.where(M_F_labels==0)
    
    #Split counts
    #total_Count=len(male[0])+len(female[0])
    m_Count=round(sample_size*(1-perc_f))
    f_Count=round(sample_size*perc_f)
    
    #Extracting data
    data_M=np.take(data,male,axis=0)[0]
    data_F=np.take(data,female,axis=0)[0]
    
    #Random selection index
    sample_index_M=np.random.choice(len(data_M), int(m_Count))
    sample_index_F=np.random.choice(len(data_F), int(f_Count))
    data_M=np.take(data_M,sample_index_M,axis=0)
    data_F=np.take(data_F,sample_index_F,axis=0)
    rebalanced_dataset=torch.cat((data_M,data_F),axis=0)
    
    #Newlabels
    label_M=torch.ones(len(data_M))
    label_F=torch.zeros(len(data_F))
    rebalanced_labels=torch.cat((label_M,label_F))
    # torch.save((rebalanced_dataset,rebalanced_labels),'./test_fairness_data_%f_%f_samples_%i'%(round(perc_f,2),round(1-perc_f,2),sample_size))

    if not (os.path.exists('./real_data')):
        os.makedirs('./real_data')
        
    torch.save((rebalanced_dataset,rebalanced_labels),'./real_data/test_fairness_data_%f_%f_samples_%i'%(round(perc_f,2),round(1-perc_f,2),sample_size))

    # return appropriate split

    dataset = torch.utils.data.TensorDataset(rebalanced_dataset, rebalanced_labels)
    return dataset

if __name__=='__main__':
    #Standard test with regards to percentages
    unbias_perc=0.5
    test_gen_ref(unbias_perc,10000)
    for i in np.arange (0.1,1,0.1):
        test_gen(i,10000) #0.5 Female
        
    #Standard test with regards to Samples
#    unbias_perc=0.5
#    bias_perc=0.9
#    for i in np.arange (1000,20000,1000):
#        test_gen_ref(unbias_perc,i)
#        test_gen(bias_perc,i) #0.5 Female
