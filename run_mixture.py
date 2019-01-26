from mixture_model_mnist import mixture_model_dvae
from mixture_model_mnist import mixture_model_gsm


params = {'num_epochs': 400,
            'supervised_epochs':1200, # 0 for unsupervised
            'num_labeled_data':200,
            'batch_size': 100,
            'gaussian_dimension':20,
            'learning_rate': 0.0008,
            'gumbels' : 1,
            'N_K': (1,10), # N should stay = 1, K could change
            'eps_0':1.0,
            'anneal_rate':1e-5,
            'min_eps':0.1,
            'ST-estimator':False, # relevant for gsm
            'split_valid':True,
            'binarize':True,
            'random_seed':666,
            'save_images':True,
            'print_every':20} # put float('Inf') for no printing


#mixture_model = mixture_model_gsm.Mixture_Model(params)
mixture_model = mixture_model_dvae.Mixture_Model(params)

nll_res,acc_res = mixture_model.training_procedure()


