# Direct Optimization through Argmax for Discrete Variational Auto-Encoder 
https://arxiv.org/abs/1806.02867


discrete_vae: <br /> 
A discrete VAE model with N categorial latent variables of size K. <br />
The gradients can be estimated with gumbel-max / gumbel-softmax reparametrizations or without a reparametrization (unbiased).
[Note that the unbiased estimator works only with N=1]<br />


mixture_model_mnist: <br />
contains mixture of gaussian and discrete VAE (unsupervised or semi-supervised) [mnist] <br />

To run the above, use the scripts run_direct_gsm.py, run_mixture.py respectively. <br />

Impementation of structured-VAE is available only with direct_vae, check out the paper for more details. <br />
run_structured_cplex.py <br />
run_structured_maxflow.py 

