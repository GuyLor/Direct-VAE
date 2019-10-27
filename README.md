# Direct Optimization through $\arg \max$ for Discrete Variational Auto-Encoder


discrete_vae: discrete vae model with categorial (N x k) latent variables <br />
mixture_model_mnist: contains mixture of gaussian and discrete VAE (unsupervised or semi-supervised) [mnist] <br />

The above are optimized with direct optimization and with gumbel-softmax. <br />

To run the above, use the scripts run_direct_gsm.py, run_mixture.py respectively. <br />

Impementation of structured-VAE  is available only with direct_vae, check out the paper for more details.
run_structured_cplex.py
run_structured_maxflow.py 

