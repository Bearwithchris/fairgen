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
import losses
from clf_models import ResNet18, BasicBlock


CLF_PATH = '../results/attr_clf/model_best.pth.tar'
MULTI_CLF_PATH = '../results/multi_clf/model_best.pth.tar'


def classify_examples(model, sample_path):
    """
    classifies generated samples into appropriate classes 
    """
    model.eval()
    preds = []
    probs = []
    samples = np.load(sample_path)['x']
    n_batches = samples.shape[0] // 1000

    with torch.no_grad():
        # generate 10K samples
        for i in range(n_batches):
            x = samples[i*1000:(i+1)*1000]
            samp = x / 255.  # renormalize to feed into classifier
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

def load_dataset(config,bias=1):
    DATA_PATH = '../../data/'
    bias_split = '{}_perc{}'.format(config['bias'], config['perc'])
    if bias==0:
        path = DATA_PATH + 'celeba_{}/celeba_balanced_train_data.pt'.format(bias_split)
        # dataset = torch.load(path).float() / 127.5 - 1
        dataset = torch.load(path).float()
        train_set = torch.utils.data.TensorDataset(dataset)
    else:
        # load unbalanced data
        path = DATA_PATH + 'celeba_{}/celeba_unbalanced_train_data.pt'.format(bias_split)
        # normalize data to be [-1, 1]
        # dataset = torch.load(path).float() / 127.5 - 1
        dataset = torch.load(path).float()
        train_set = torch.utils.data.TensorDataset(dataset)
    DataLoader=torch.utils.data.DataLoader(train_set,batch_size=50,shuffle=True)
    print ("Data extracted from: %s"%path)
    return (DataLoader,len(dataset))
                
def run(config):
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
    config['sample_num_npz'] = 10000
    print(config['ema_start'])

    # Seed RNG
    # utils.seed_rng(config['seed'])  # config['seed'])

    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True



    experiment_name = (config['experiment_name'] if config['experiment_name'] #Default CelebA
                        else utils.name_from_config(config))
    
    #Create Bias Datasets in npz forma
    bias=1
    train_set,size=load_dataset(config,bias)
    
    # classify examples and get probabilties
    n_classes = 2
    if config['multi']:
        n_classes = 4
   
    print ("Preparing data....")
    print ("Dataset has a total of %i data instances"%size)
    k=0
    num_npz=4
    npz_filename = '%s/%s/fid_real_samples_%s.npz' % (config['samples_root'], experiment_name, k) #E.g. fid_samples_0
        
    if os.path.exists(npz_filename):
        print('samples already exist, skipping...')
        pass
    else:
        X = []
        pbar = tqdm(train_set)
        print('Sampling %d images and saving them to npz...' %config['sample_num_npz']) #10k
        count=0
        for i ,x in enumerate(pbar):
            #Cap number of Npz files generated
            if num_npz<k:
                break
            
            if count<config['sample_num_npz']/50: #Train by batch
                X+=x
                count+=1

                
            else:
                #Save old file
                # X = np.concatenate(X, 0)[:config['sample_num_npz']]
                # X = np.array(X)]
                X=np.array(torch.vstack(X)).astype(np.uint8)
                npz_filename = '%s/%s/fid_real_samples_%s.npz' % (config['samples_root'], experiment_name, k)
                print('Saving %i counts of filetype npz to %s...' %(count, npz_filename))
                np.savez(npz_filename, **{'x': X})
                
                #Reset counter and array
                X = []
                count=0
                
                #Start new file
                k+=1
                npz_filename = '%s/%s/fid_real_samples_%s.npz' % (config['samples_root'], experiment_name, k) #E.g. fid_samples_0
                
    
    #=====Classify===================================================================
    metrics = {'l2': 0, 'l1': 0, 'kl': 0}
    l2_db = np.zeros(10)
    l1_db = np.zeros(10)
    kl_db = np.zeros(10)

    # output file
    fname = '%s/%s/fair_disc_fid_samples.p' % (config['samples_root'], experiment_name)

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

    #Initialise datalogger
    data_logger=open("E:\\GIT\\fairgen_original\\datalogger.txt","a")
    data_logger.write("Experiment: %s, bias=%i, data= 1 \n"%(config['bias'],bias))
    
    # classify examples and get probabilties
    n_classes = 2
    if config['multi']:
        n_classes = 4

    # number of classes
    probs_db = np.zeros((5, 10000, n_classes)) #Numper of runs , images per run ,Number of classes
    for i in range(5):
        # grab appropriate samples
        npz_filename = '{}/{}/{}_real_samples_{}.npz'.format(config['samples_root'], experiment_name, config['mode'], i)
        print ("Loading data from: %s"%npz_filename)
        preds, probs = classify_examples(clf, npz_filename) #Classify the data
        
        # for i in preds:
        #     print (i)
        # time.sleep(100)
        
        l2, l1, kl, ce = utils.fairness_discrepancy(preds, clf_classes) #Pass to calculate score
        
        #exp
        l2Exp, l1Exp, klExp ,ceExp = utils.fairness_discrepancy_exp(probs, clf_classes) #Pass to calculate score

        # save metrics (To add on new mertrics add here)
        l2_db[i] = l2
        l1_db[i] = l1
        kl_db[i] = kl
        probs_db[i] = probs
        print('fair_disc for iter {} is: l2:{}, l1:{}, kl:{}, ce:{}'.format(i, l2, l1, kl,ce))
        print('fair_disc_exp for iter {} is: l2:{}, l1:{}, kl:{}, ce:{} \n'.format(i, l2Exp, l1Exp, klExp,ceExp))
        
        #Log the results
        data_logger.write(('fair_disc for iter {} is: l2:{}, l1:{}, kl:{}, ce:{} \n'.format(i, l2, l1, kl,ce)))
        data_logger.write(('fair_disc_exp for iter {} is: l2:{}, l1:{}, kl:{}, ce:{} \n'.format(i, l2Exp, l1Exp, klExp,ceExp)))
        
        
    metrics['l2'] = l2_db
    metrics['l1'] = l1_db
    metrics['kl'] = kl_db
    print('fairness discrepancies saved in {}'.format(fname))
    print(l2_db)
    data_logger.close()

    # save all metrics
    with open(fname, 'wb') as fp:
        pickle.dump(metrics, fp)
    np.save(os.path.join(config['samples_root'], experiment_name, 'clf_probs.npy'), probs_db)


def main():
    # parse command line and run
    parser = utils.prepare_parser()
    parser = utils.add_sample_parser(parser)
    config = vars(parser.parse_args())
    print(config)
    run(config)


if __name__ == '__main__':
    main()
