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

def load_dataset(config,bias=0):
    DATA_PATH = '../../data/'
    bias_split = '{}_perc{}'.format(config['bias'], config['perc'])
    if bias==0:
        balanced_path = DATA_PATH + 'celeba_{}/celeba_balanced_train_data.pt'.format(bias_split)
        dataset = torch.load(balanced_path).float() / 127.5 - 1
        train_set = torch.utils.data.TensorDataset(dataset)
    else:
        # load unbalanced data
        unbalanced_path = DATA_PATH + 'celeba_{}/celeba_unbalanced_train_data.pt'.format(bias_split)
        unbalanced_labels_path = DATA_PATH + 'celeba_{}/celeba_unbalanced_train_labels.pt'.format(bias_split)
        # normalize data to be [-1, 1]
        dataset = torch.load(unbalanced_path).float() / 127.5 - 1
        train_set = torch.utils.data.TensorDataset(dataset)
    return (train_set,len(dataset))
                
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

#Removal Of Generator Module Start=======================================================

    # # Import the model--this line allows us to dynamically select different files.
    # model = __import__(config['model']) #Default BigGAN
    # experiment_name = (config['experiment_name'] if config['experiment_name'] #Default CelebA
    #                     else utils.name_from_config(config))
    # print('Experiment name is %s' % experiment_name)

    # #Pass generator to GPU 
    # G = model.Generator(**config).cuda() #BIGGAN.generator()
    # utils.count_parameters(G)

    # # Load weights
    # print('Loading weights...')
    # assert config['mode'] in ['fair', 'fid']
    # print('sampling from model with best FID scores...')
    # config['mode'] = 'fid'  # can change this to 'fair', but this assumes access to ground-truth attribute classifier (and labels)
  
    
    # # find best weights for either FID or fair checkpointing (Load best m weights)
    # weights_root = config['weights_root']
    # ckpts = glob.glob(os.path.join(weights_root, experiment_name, 'state_dict_best_{}*'.format(config['mode'])))
    # best_ckpt = 'best_{}{}'.format(config['mode'],len(ckpts)-1)
    # config['load_weights'] = best_ckpt

    # # load weights to sample from generator
    # utils.load_weights(G if not (config['use_ema']) else None, None, state_dict, weights_root, experiment_name, config['load_weights'], G if config['ema'] and config['use_ema'] else None,
    #     strict=False, load_optim=False)

    # # Update batch size setting used for G
    # G_batch_size = max(config['G_batch_size'], config['batch_size'])
    
    # #Create a object with the appopriate dimension for samples to be generated
    # z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
    #                             device=device, fp16=config['G_fp16'],
    #                             z_var=config['z_var'])
    # print('Putting G in eval mode..')
    # G.eval()

    # # Sample function (Tools to generate samples with vector Z and Y embedding)
    # sample = functools.partial(utils.sample, G=G, z_=z_, y_=y_, config=config)
    # if config['accumulate_stats']:
    #     print('Accumulating standing stats across %d accumulations...' %
    #           config['num_standing_accumulations'])
    #     utils.accumulate_standing_stats(G, z_, y_, config['n_classes'],
    #                                     config['num_standing_accumulations'])

    # # # # Sample a number of images and save them to an NPZ, for use with TF-Inception

    
    # #Create sample directory
    # sample_path = '%s/%s/' % (config['samples_root'], experiment_name)
    # print('looking in sample path {}'.format(sample_path))
    # if not os.path.exists(sample_path):
    #     print('creating sample path: {}'.format(sample_path))
    #     os.mkdir(sample_path)
    
    # #(Generating sample images)===========================================================
    # # Lists to hold images and labels for images e.g. "celeba_90_10_perc1.0_impweight"
    # print('saving samples from best FID checkpoint')
    
    # # sampling 10 sets of 10K samples
    # for k in range(10):
    #     npz_filename = '%s/%s/fid_samples_%s.npz' % (config['samples_root'], experiment_name, k) #E.g. fid_samples_0
    #     if os.path.exists(npz_filename):
    #         print('samples already exist, skipping...')
    #         continue
    #     x, y = [], []
    #     print('Sampling %d images and saving them to npz...' %
    #           config['sample_num_npz']) #10k
    #     for i in trange(int(np.ceil(config['sample_num_npz'] / float(G_batch_size)))):
    #         with torch.no_grad():
    #             images, labels = sample()
    #         #save the image and label
    #         x += [np.uint8(255 * (images.cpu().numpy() + 1) / 2.)]
    #         y += [labels.cpu().numpy()]
        
    #     #Not too sure the purpose of this atm
    #     x = np.concatenate(x, 0)[:config['sample_num_npz']]
    #     y = np.concatenate(y, 0)[:config['sample_num_npz']]
    #     print('checking labels: {}'.format(y.sum()))
    #     print('Images shape: %s, Labels shape: %s' % (x.shape, y.shape))
    #     npz_filename = '%s/%s/fid_samples_%s.npz' % (config['samples_root'], experiment_name, k)
    #     print('Saving npz to %s...' % npz_filename)
    #     np.savez(npz_filename, **{'x': x, 'y': y})

#Removal Of Generator Module Start End======================================================

    experiment_name = (config['experiment_name'] if config['experiment_name'] #Default CelebA
                        else utils.name_from_config(config))
    
    #Create Bias Datasets in npz format
    train_set,size=load_dataset(config,bias=1)
    
    # classify examples and get probabilties
    n_classes = 2
    if config['multi']:
        n_classes = 4
   
    print ("Preparing data....")
    print ("Dataset has a total of %i data instances"%size)
    k=1
    num_npz=10
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
            
            if count<config['sample_num_npz']:
                X+=x
                count+=1

                
            else:
                print (len(X))
                # time.sleep(10)
                #Save old file
                X = np.concatenate(x, 0)[:config['sample_num_npz']]
                npz_filename = '%s/%s/fid_real_samples_%s.npz' % (config['samples_root'], experiment_name, k)
                print('Saving npz to %s...' % npz_filename)
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

    # classify examples and get probabilties
    n_classes = 2
    if config['multi']:
        n_classes = 4

    # number of classes
    probs_db = np.zeros((5, 10000, n_classes)) #Numper of runs , images per run ,Number of classes
    for i in range(5):
        # grab appropriate samples
        npz_filename = '{}/{}/{}_samples_{}.npz'.format(config['samples_root'], experiment_name, config['mode'], i)
        preds, probs = classify_examples(clf, npz_filename) #Classify the data
        l2, l1, kl = utils.fairness_discrepancy(preds, clf_classes) #Pass to calculate score
        
        #exp
        l2Exp, l1Exp, klExp = utils.fairness_discrepancy_exp(probs, clf_classes) #Pass to calculate score

        # save metrics (To add on new mertrics add here)
        l2_db[i] = l2
        l1_db[i] = l1
        kl_db[i] = kl
        probs_db[i] = probs
        print('fair_disc for iter {} is: l2:{}, l1:{}, kl:{}'.format(i, l2, l1, kl))
        print('fair_disc_exp for iter {} is: l2:{}, l1:{}, kl:{} \n'.format(i, l2Exp, l1Exp, klExp))
        
    metrics['l2'] = l2_db
    metrics['l1'] = l1_db
    metrics['kl'] = kl_db
    print('fairness discrepancies saved in {}'.format(fname))
    print(l2_db)
    
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
