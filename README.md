# direct_vae


discrete_vae: discrete vae model with categorial (1 x k) latent variables <br />
mixture_model_mnist: contains mixture of gaussian and discrete vae (unsupervised or semi-supervised) [mnist] <br />

The above are optimized with direct optimization and with gumbel-softmax. <br />

REBAR, RELAX, ARM:  only for binary latent variables, based on https://github.com/duvenaud/relax/blob/master/mnist_vae.py and https://github.com/mingzhang-yin/ARM-gradient

To run the above, use the scripts run_direct_gsm.py, run_mixture.py, run_relax_rebar.py, run_ARM.py respectively. <br />

In addition, an impementation of structured vae is available only with direct_vae, check out the paper for more details. run_structured_dvae.py 

