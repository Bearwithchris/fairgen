''' Sample
   This script loads a pretrained net and a weightsfile and sample '''
import time
import os
import glob
import sys

import functools
import math
import numpy as np
from tqdm import tqdm, trange
import pickle
from argparse import ArgumentParser


import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision

# Import my stuff
import inception_utils
import utils
import utils_add_on as uao
import losses
from clf_models import ResNet18, BasicBlock
import fid_score_mod

CLF_PATH = '../../results/attr_clf/model_best.pth.tar'
MULTI_CLF_PATH = '../../results/multi_clf/model_best.pth.tar'


def classify_examples(model, sample_path):
    """
    classifies generated samples into appropriate classes 
    """
    model.eval()
    preds = []
    probs = []
    samples = np.load(sample_path)['x']
    n_batches = samples.shape[0] // 1000
    print (sample_path)

    with torch.no_grad():
        # generate 10K samples
        for i in range(n_batches):
            x = samples[i*1000:(i+1)*1000]
            samp = x  # renormalize to feed into classifier
            samp = torch.from_numpy(samp).to('cuda').float()

            # get classifier predictions
            logits, probas = model(samp)
            _, pred = torch.max(probas, 1) #Returns the max indices i.e. index
            probs.append(probas)
            preds.append(pred)
        preds = torch.cat(preds).data.cpu().numpy()
        probs = torch.cat(probs).data.cpu().numpy()
        # probs = torch.cat(probs).data.cpu()

    return preds, probs

def run(config):
    sample_path="./toy_samples"
    
    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num_fair': 0, 'save_best_num_fid': 0, 'best_IS': 0, 'best_FID': 999999, 'best_fair_d': 999999, 'config': config}

    # Optionally, get the configuration from the state dict. This allows for
    # recovery of the config provided only a state dict and experiment name,
    # and can be convenient for writing less verbose sample shell scripts.
    
    if config['config_from_name']: #Default is false 
        utils.load_weights(None, None, state_dict, config['weights_root'],
                           config['experiment_name'], config['load_weights'], None,
                           strict=False, load_optim=False)
        # Ignore items which we might want to overwrite from the command line
        for item in state_dict['config']:
            if item not in ['z_var', 'base_root', 'batch_size', 'G_batch_size', 'use_ema', 'G_eval_mode']:
                config[item] = state_dict['config'][item]

    # update config (see train.py for explanation)
    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = 1
    
    if config['conditional']:
        config['n_classes'] = 2
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    config = utils.update_config_roots(config)
    config['skip_init'] = True
    config['no_optim'] = True
    device = 'cuda'
    config['sample_num_npz'] = config['sampleC'] # To Edit
    perc_bias=float(config["bias"].split("_")[0])/100
    print(config['ema_start'])

    # Seed RNG
    # utils.seed_rng(config['seed'])  # config['seed'])

    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True


    #Log Runs
    f=open('%s/log_sample_exp.txt' %(sample_path),"a")

    experiment_name = (config['experiment_name'] if config['experiment_name'] #Default CelebA
                        else utils.name_from_config(config))
    
    #Create Bias Datasets in npz format
    train_set,size,labels=uao.load_data_toy(perc_bias,config['sample_num_npz'])
    train_set_ref,size_ref,labels_ref=uao.load_data_toy_ref(perc_bias,config['sample_num_npz'])
    
    # classify examples and get probabilties
    n_classes = 2
    if config['multi']:
        n_classes = 4
   
    print ("Preparing data....")
    print ("Dataset has a total of %i data instances"%size)   
    
    #Prepare Reference Data*******************************************************************************************
    k=0
    

    npz_filename_ref = '%s/%s_fid_real_ref_samples_%s.npz' % (sample_path, perc_bias, k) #E.g. perc_fid_samples_0
    if os.path.exists(npz_filename_ref):
        print('samples already exist, skipping...')
        #pass
    else:
        X = []
        pbar = tqdm(train_set_ref)
        print('Sampling %d images and saving them to npz...' %config['sample_num_npz']) #10k
        count=1 
        
        for i ,x in enumerate(pbar):
            X+=x                
        X=np.array(torch.stack(X)).astype(np.uint8)
        print('Saving npz to %s...' % npz_filename_ref)
        # print(X.shape)
        # time.sleep(100)
        np.savez(npz_filename_ref, **{'x': X})
    
    
    #Preparing Comparison Data*******************************************************************************************
    k=0
    

    npz_filename = '%s/%s_fid_real_samples_%s.npz' % (sample_path, perc_bias, k) #E.g. perc_fid_samples_0
    if os.path.exists(npz_filename):
        print('samples already exist, skipping...')
        #pass
    else:
        X = []
        pbar = tqdm(train_set)
        print('Sampling %d images and saving them to npz...' %config['sample_num_npz']) #10k
        count=1 
        
        for i ,x in enumerate(pbar):
            X+=x                
        X=np.array(torch.stack(X)).astype(np.uint8)
        print('Saving npz to %s...' % npz_filename)
        # print(X.shape)
        # time.sleep(100)
        np.savez(npz_filename, **{'x': X})
                
   

    #=====Classify===================================================================
    metrics = {'l2': 0, 'l1': 0, 'kl': 0}
    l2_db = np.zeros(10)
    l1_db = np.zeros(10)
    kl_db = np.zeros(10)

    # output file
    fname = '%s/%s_fair_disc_fid_samples.p' % (sample_path, perc_bias)

    # load classifier 
    #(Saved state)
    if not config['multi']:
        print('Pre-loading pre-trained single-attribute classifier...')
        clf_state_dict = torch.load(CLF_PATH)['state_dict']
        clf_classes = 2
    else:
        # multi-attribute
        print('Pre-loading pre-trained multi-attribute classifier...')
        clf_state_dict = torch.load(MULTI_CLF_PATH)['state_dict']
        clf_classes = 4
        
    # load attribute classifier here
    #(Model itself)
    clf = ResNet18(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=clf_classes, grayscale=False) 
    clf.load_state_dict(clf_state_dict)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clf = clf.to(device)
    clf.eval()  # turn off batch norm

    # classify examples and get probabilties
    n_classes = 2
    if config['multi']:
        n_classes = 4

    # number of classes
    probs_db = np.zeros((1, config['sample_num_npz'], n_classes)) #Numper of runs , images per run ,Number of classes
    for i in range(1):
        # grab appropriate samples
        npz_filename = '%s/%s_fid_real_samples_%s.npz' % (sample_path, perc_bias, k) #E.g. perc_fid_samples_0
        
        # preds, probs = classify_examples(clf, npz_filename) #Classify the data

        # l2, l1, kl = utils.fairness_discrepancy(preds, clf_classes) #Pass to calculate score
        
        # #exp
        # l2Exp, l1Exp, klExp = utils.fairness_discrepancy_exp(probs, clf_classes) #Pass to calculate score

        # # save metrics (To add on new mertrics add here)
        # l2_db[i] = l2
        # l1_db[i] = l1
        # kl_db[i] = kl
        # probs_db[i] = probs
        
        # #Write log
        # f.write("Running: "+npz_filename+"\n")
        # f.write('fair_disc for iter {} is: l2:{}, l1:{}, kl:{} \n'.format(i, l2, l1, kl))
        
        
        # print('fair_disc for iter {} is: l2:{}, l1:{}, kl:{}'.format(i, l2, l1, kl))
        # print('fair_disc_exp for iter {} is: l2:{}, l1:{}, kl:{} \n'.format(i, l2Exp, l1Exp, klExp))
        
        #FID score 50_50 vs others 
        # data_moments=os.path.join(sample_path,"0.5_fid_real_samples_0.npz")
        data_moments=os.path.join(sample_path,"0.1_fid_real_samples_ref_0.npz")
        sample_moments=os.path.join(sample_path,'%s_fid_real_samples_%s.npz'%(perc_bias,k))
        FID = fid_score_mod.calculate_fid_given_paths([data_moments, sample_moments], batch_size=100, cuda=True, dims=2048)
        print ("FID_Fair: "+str(FID))
        f.write("Bias: "+str(perc_bias)+ " FID_fair: "+str(FID)+"\n")      
        f.close()
    metrics['l2'] = l2_db
    metrics['l1'] = l1_db
    metrics['kl'] = kl_db
    print('fairness discrepancies saved in {}'.format(fname))
    print(l2_db)
    
    # save all metrics
    with open(fname, 'wb') as fp:
        pickle.dump(metrics, fp)
    np.save(os.path.join(config['samples_root'], 'clf_probs.npy'), probs_db)


def main():
    # parse command line and run
    parser = utils.prepare_parser()
    parser = utils.add_sample_parser(parser)
    parser = utils.add_sample_parser_exp(parser)
    config = vars(parser.parse_args())
    print(config)
    run(config)


if __name__ == '__main__':
    main()
