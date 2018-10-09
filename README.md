# direct_vae

Implementation of direct optimization paper.

discrete_vae: contains discrete vae model [mnist, omniglot datasets]
mixture_model_mnist: contains mixture of gaussian and discrete vae (unsupervised or semi-supervised) [mnist]

The above are optimized with direct optimization and with gumbel-softmax.

REBAR_RELAX: this implementation is only with binary latent variables, based on https://github.com/duvenaud/relax/blob/master/mnist_vae.py

To run the above, use the scripts run_direct_gsm.py, run_mixture.py, run_relax_rebar.py respectively.

Dependencies:
Python 2.7
Pytorch 0.4.0
(REBAR_RELAX: Tensorflow)

