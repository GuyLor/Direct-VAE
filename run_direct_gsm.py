import torch
from discrete_vae import DVAE
from discrete_vae import GSM
from discrete_vae import unbiased

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--method', type=str, default='direct',
                    help='direct, gsm or unbiased method')
parser.add_argument('--ds', type=str, default='mnist',
                    help='mnist, fashion-mnist or omniglot')
parser.add_argument('--seed', type=int, default=775,
                    help='random seed')
parser.add_argument('--N', type=int, default=1,
                    help='dimension size of the latent space z (number of rows)')
parser.add_argument('--K', type=int, default=10,
                    help='dimension size of the latent space z (number of columns)')

args = parser.parse_args()

params = {'num_epochs': 300,
            'composed_decoder': True,
            'batch_size': 100,
            'learning_rate': 0.001,
            'gumbels' : 1,
            'N_K': (args.N,args.K),
            'eps_0':1.0,
            'anneal_rate':1e-5,
            'unbiased':True,
            'min_eps':0.1,
            'random_seed':args.seed,
            'dataset':args.ds, # 'mnist' or 'fashion-mnist' or 'omniglot'
            'split_valid':True,
            'binarize':True,
            'ST-estimator':False, # relevant only for GSM
            'save_images':False,
            'print_result':True}




"""
returned results:

 train_results: list, where the i'th element is the average nll of the mini-batches of i'th epoch,
 test_results: list, where the i'th element is the average nll of the mini-batches of i'th epoch,
 best_test_nll: the test nll of the epoch where the validation nll is the best of all epochs,
 best_state_dicts: pytorch models,
 params
 """

if args.method == 'direct':
    params['unbiased'] = False
    dvae_results = DVAE.training_procedure(params)
if args.method == 'gsm':
    params['unbiased'] = False
    gsm_results = GSM.training_procedure(params)
if args.method == 'unbiased':
    params['unbiased'] = True
    unbiased_results = unbiased.training_procedure(params)


