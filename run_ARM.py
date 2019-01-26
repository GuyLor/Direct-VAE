from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import datas

import sys

import scipy.stats as stats
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

params = {'binarize':True,
          'dataset':'mnist',
          'random_seed':777,
          'split_valid':False,
          'batch_size':100}
batch_size = params['batch_size']
train_loader,test_loader = datas.load_data(params)

slim=tf.contrib.slim
Bernoulli = tf.contrib.distributions.Bernoulli

#%%
def bernoulli_loglikelihood(b, log_alpha):
    return b * (-tf.nn.softplus(-log_alpha)) + (1 - b) * (-log_alpha - tf.nn.softplus(-log_alpha))

def lrelu(x, alpha=0.1):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)



def encoder(x,b_dim,reuse=False):
    with tf.variable_scope("encoder", reuse = reuse):
        #h2 = slim.stack(x, slim.fully_connected,[200,200],activation_fn=lrelu)
        h2 = tf.layers.dense(x, 300, tf.nn.relu, name="encoder_1",use_bias=True)
        log_alpha = tf.layers.dense(h2, b_dim,use_bias=True)
    return log_alpha


def decoder(b,x_dim,reuse=False):
    #return logits
    with tf.variable_scope("decoder", reuse = reuse):
        #h2 = slim.stack(b ,slim.fully_connected,[200,200],activation_fn=lrelu)
        h2 = tf.layers.dense(b, 300, tf.nn.relu, name="decoder_1",use_bias=True)
        log_alpha = tf.layers.dense(h2, x_dim,use_bias=True)
    return log_alpha


def fun1(x_star,log_alpha_b,E,axis_dim=1,reuse_encoder=False,reuse_decoder=False):
    '''
    x_star,E are N*(d_x or d_b)
    calculate log p(x_star|E) + log p(E) - log q(E|x_star)
    axis_dim is axis for d_x or d_b
    x_star is observe x; E is latent b
    return elbo,  N*K
    '''
    #log q(E|x_star), b_dim is global
    #log_alpha_b = encoder(x_star,b_dim,reuse=reuse_encoder)
    log_q_b_given_x = bernoulli_loglikelihood(E, log_alpha_b)
    # (N,K),conditional independent d_b Bernoulli
    log_q_b_given_x = tf.reduce_sum(log_q_b_given_x , axis=axis_dim)

    #log p(E)
    log_p_b = bernoulli_loglikelihood(E, tf.zeros_like(E))
    log_p_b = tf.reduce_sum(log_p_b, axis=axis_dim)
    
    #log p(x_star|E), x_dim is global
    log_alpha_x = decoder(E,x_dim,reuse=reuse_decoder)
    log_p_x_given_b = bernoulli_loglikelihood(x_star, log_alpha_x)
    log_p_x_given_b = tf.reduce_sum(log_p_x_given_b, axis=axis_dim)
    
    # neg-ELBO
    return log_q_b_given_x - (log_p_x_given_b + log_p_b)

def evidence(sess,elbo, batch_size = 100, S = 100, total_batch = None):
    '''
    For correct use:
    ELBO for x_i must be calculated by SINGLE z sample from q(z|x_i)
    '''
    #from scipy.special import logsumexp
    if total_batch is None:
        total_batch = int(data.num_examples / batch_size)

    avg_evi = 0

    for np_xtest,_ in test_loader:
        elbo_i = sess.run(elbo,{x:np_xtest.numpy()})
        avg_evi += np.mean(elbo_i)
    avg_evi= avg_evi/ len(test_loader)
    return avg_evi

tf.reset_default_graph()

b_dim = 15; x_dim = 784
eps = 1e-10

lr=tf.constant(0.001)

x = tf.placeholder(tf.float32,[None,x_dim]) #N*d_x
x_binary = tf.to_float(x > .5)

N = tf.shape(x_binary)[0]

#logits for bernoulli, encoder q(b|x) = log Ber(b|log_alpha_b)
log_alpha_b = encoder(x_binary,b_dim)  #N*d_b

q_b = Bernoulli(logits=log_alpha_b) #sample K_b \bv
b_sample = tf.cast(q_b.sample(),tf.float32) #K_b*N*d_b, accompanying with encoder parameter, cannot backprop

#compute decoder p(x|b), gradient of decoder parameter can be automatically given by loss
neg_elbo = fun1(x_binary,log_alpha_b,b_sample,reuse_encoder=True,reuse_decoder= False)[:,np.newaxis]
gen_loss = tf.reduce_mean(neg_elbo) #average over N


gen_opt = tf.train.AdamOptimizer(lr)
gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')
gen_gradvars = gen_opt.compute_gradients(gen_loss, var_list=gen_vars)
gen_train_op = gen_opt.apply_gradients(gen_gradvars)

#provide encoder q(b|x) gradient by data augmentation
u_noise = tf.random_uniform(shape=[N,b_dim],maxval=1.0) #sample K \uv
P1 = tf.sigmoid(-log_alpha_b)
E1 = tf.cast(u_noise>P1,tf.float32)
P2 = 1 - P1
E2 = tf.cast(u_noise<P2,tf.float32)


F1 = fun1(x_binary,log_alpha_b,E1,axis_dim=1,reuse_encoder=True,reuse_decoder=True)
F2 = fun1(x_binary,log_alpha_b,E2,axis_dim=1,reuse_encoder=True,reuse_decoder=True)

alpha_grads = tf.expand_dims(F1-F2,axis=1)*(u_noise-0.5) #N*d_b
#alpha_grads = tf.reduce_mean(alpha_grads,axis=1) #N*d_b
inf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')
#log_alpha_b is N*d_b, alpha_grads is N*d_b, inf_vars is d_theta
inf_grads = tf.gradients(log_alpha_b, inf_vars, grad_ys=alpha_grads)
inf_gradvars = zip(inf_grads, inf_vars)
inf_opt = tf.train.AdamOptimizer(lr)
inf_train_op = inf_opt.apply_gradients(inf_gradvars)


with tf.control_dependencies([gen_train_op, inf_train_op]):
    train_op = tf.no_op()

init_op=tf.global_variables_initializer()
def get_loss(sess,data,total_batch):
    cost_eval = []
    for j in range(total_batch):
        xs = data.next_batch(batch_size)
        cost_eval.append(sess.run(neg_elbo,{x:xs}))
    return np.mean(cost_eval)

directory = os.getcwd()+'/discrete_out/'
if not os.path.exists(directory):
    os.makedirs(directory)

np_lr = 0.001
EXPERIMENT = 'MNIST_Bernoulli_ARM' + '_non_'+str(int(np.random.randint(0,100,1)))
print('Training starts....',EXPERIMENT)
print('Learning rate....',np_lr)

sess=tf.InteractiveSession()
sess.run(init_op)
step = 0


losses=[]
NUM_EPOCHS = 300
for epoch in range(1,NUM_EPOCHS+1):
    train_loss = 0
    test_loss = 0
    for np_x,np_y in train_loader:
        #i += 1

        _,np_loss=sess.run([train_op,gen_loss],{x:np_x.numpy(),lr:np_lr})
        
        train_loss += np_loss


    test_loss=evidence(sess, neg_elbo, batch_size, S = 100, total_batch=10)


    train_loss /= len(train_loader)
    #test_loss /= len(test_loader)
    losses.append([train_loss,test_loss])
    print('epoch %d, train ELBO: %0.3f, test ELBO: %0.3f ' % (epoch,train_loss, test_loss))
#import torch
#torch.save(losses,'tf_ARM_15_binary_2_seed666.tar')
