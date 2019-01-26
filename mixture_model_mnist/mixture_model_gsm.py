
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
import torchvision

import math
import random
import numpy as np

import datas
from utils import *


def sample_gumbel(shape, eps=1e-20):
        #Sample from Gumbel(0, 1)
        U = torch.rand(shape).float()
        return - torch.log(eps - torch.log(U + eps))

def gumbel_softmax_sample(logits, tau=1, eps=1e-20):

    dims = len(logits.size())
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    y = logits + to_var(gumbel_noise)
    return F.softmax(y / tau,dim =-1)

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):

    shape = logits.size()
    assert len(shape) == 2
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)

    if hard:
        _, k = y_soft.max(-1)

        y_hard = torch.FloatTensor(*shape).zero_().scatter_(-1, k.view(-1, 1).cpu(), 1.0)

        y = to_var(y_hard) - to_var(y_soft.detach()) + y_soft
    else:
        y = y_soft

    return y
"""
def paded_mm(one_hot_matrix,other_matrix): # one_hot_matrix.size: (batch,N,K), other_matrix: (batch,K,gauss_dim)
    one_hots = one_hot_matrix.transpose(1,2).contiguous()
    o = to_var(torch.ones(one_hot_matrix.size(0),one_hot_matrix.size(1),other_matrix.size(2)))
    return torch.bmm(one_hots,o)*other_matrix
"""
def paded_mm(one_hot_matrix,other_matrix):
    # one_hot_matrix.size: (batch,N,K), other_matrix: (batch,K,gauss_dim)
    N ,K= one_hot_matrix.size(1),one_hot_matrix.size(2)
    gaussian_dim = other_matrix.size(2)
    one_hot_matrix = one_hot_matrix.view(-1,N,K)
    other_matrix = other_matrix.view(-1,K,gaussian_dim)
    one_hots = one_hot_matrix.transpose(1,2).contiguous()
    o = to_var(torch.ones(one_hot_matrix.size(0),one_hot_matrix.size(1),other_matrix.size(2)))
    return torch.bmm(one_hots,o)*other_matrix
# VAE model
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim= 400, gaussian_dim= 20 ,N = 1, K = 10, D = 15):
        super(VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(image_size,int(h_dim)),
            nn.LeakyReLU(0.2),
            nn.Linear(int(h_dim), h_dim/2))

        self.encode_x_to_gibbs = nn.Sequential(
            nn.Linear(h_dim/2, h_dim/4),
            nn.LeakyReLU(0.2),
            nn.Linear(h_dim/4, N*K))

        self.encode_x_to_gauss = nn.Sequential(
            nn.Linear(h_dim/2,h_dim/4 * K))

        self.encode_K_gauss = nn.Sequential(
            nn.Linear(h_dim/4 ,h_dim/6),
            nn.LeakyReLU(0.5),
            nn.Dropout(0.5),
            nn.Linear(h_dim/6, 2*gaussian_dim)) # 1 for mu 1 for log_var
	"""
        self.encode_x_to_gauss = nn.Sequential(
            nn.Linear(h_dim/2,gaussian_dim * K))
            
        self.encode_K_gauss = nn.Sequential(
            nn.Linear(gaussian_dim ,int(1.5*gaussian_dim)),
            nn.LeakyReLU(0.5),
            nn.Dropout(0.5),
            nn.Linear(int(1.5*gaussian_dim), 2*gaussian_dim)) # 1 for mu 1 for log_var
	"""
        self.decoder = nn.Sequential(
            nn.Linear(gaussian_dim*K, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, image_size))
            
        self.N, self.K, self.D = N, K, D
        self.gaussian_dim = gaussian_dim
    
    
    def reparametrize(self, mu, log_var):
        """"z = mean + eps * sigma where eps is sampled from N(0, 1)."""
        eps = to_var(torch.randn(mu.size(0), mu.size(1)))
        z = mu + eps * torch.exp(log_var/2)    # 2 for convert var to std
        return z
            
    def forward(self, x,tau,hard):
        N, K, D = self.N, self.K, self.D
        gaussian_dim = self.gaussian_dim
        
        x = self.encoder(x)
        gibbs = self.encode_x_to_gibbs(x).view(-1,K)

        phi_x_ = duplicate_along(gibbs,D).view(-1,K)
        z_d = gumbel_softmax(phi_x_,tau,hard=hard).view(-1,N,D,K)
        z_discrete = torch.mean(z_d,dim=2)
        
        h = self.encode_x_to_gauss(x)
        h = h.view(h.size(0)*K,-1)
        gauss_latent = self.encode_K_gauss(h)
        mu,logs = torch.chunk(gauss_latent, 2, dim=1)
        mues = mu.contiguous()
        logs = logs.contiguous()
        z_c  = self.reparametrize(mu,logs)
        z_continuous = z_c.view(-1,K,gaussian_dim)
        
        mues = mues.view(-1,K,gaussian_dim)
        logs = logs.view(-1,K,gaussian_dim)
        #paded needs: one_hot_matrix.size: (batch,N,K), other_matrix: (batch,K,gauss_dim)
        mixture = paded_mm(z_discrete,z_continuous).view(-1,K*gaussian_dim)
        mues = torch.bmm(z_discrete,mues.contiguous().view(-1,K,gaussian_dim)).view(-1,gaussian_dim)
        log_var = torch.bmm(z_discrete,logs.contiguous().view(-1,K,gaussian_dim)).view(-1,gaussian_dim)
        out = self.decoder(mixture)
        return out,mues, log_var,gibbs, z_discrete
    
    def sample(self, gibbs, z):
        gauss = z.view(-1,self.K,self.gaussian_dim)
        mixture = paded_mm(gibbs,gauss)
        mixture = mixture.view(-1,self.K*self.gaussian_dim)
        return self.decoder(mixture)


class Mixture_Model:
    def __init__(self,params):
        self.N,self.K=params['N_K']    # multinomial with K modes
        self.D = params['gumbels']  # average of D Gumbel variables
        self.gaussian_dim = params['gaussian_dimension']
        self.params = params
        
        self.num_epochs = params['num_epochs']
        self.epoch_semi_sup = params['supervised_epochs']
        self.num_epochs += self.epoch_semi_sup
        batch_size = params['batch_size']

        params['dataset'] = 'mnist'
        self.train_loader,self.valid_loader,self.test_loader = datas.load_data(params)
        
        if self.epoch_semi_sup>0:
            train_ds,_ = datas.get_pytorch_mnist_datasets()
            balanced_ds = datas.get_balanced_dataset(train_ds,params['num_labeled_data'])
            self.train_loader_balanced = torch.utils.data.DataLoader(dataset=balanced_ds,
                                                                     batch_size=batch_size,
                                                                     shuffle=True)
        H = 400
        self.tau = params['eps_0']
        self.vae = VAE(h_dim=H,N=self.N,K=self.K,D=self.D)
        print (self.vae)
        print ('number of parameters: ', sum(param.numel() for param in self.vae.parameters()))
        print (params)
        if torch.cuda.is_available():
            self.vae.cuda()
        #vae.load_state_dict(torch.load('vae_multi_gauss_new.pkl',lambda storage, loc: storage)) ]
        self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=params['learning_rate'])
        self.print_every = params['print_every']
    
    def train(self,data_loader,semi):
        bce,kl,nll =0,0,0
        for i, (images, labels) in enumerate(data_loader):
        
	    labels = to_var(labels)        
            images = to_var(images.view(images.size(0), -1))
            out,mu, log_var,gibbs,g_softmax = self.vae(images,self.tau,hard=self.params['ST-estimator'])
            # Compute reconstruction loss and kl divergence
            reconst_loss = F.binary_cross_entropy_with_logits(out, images,reduction='sum')
            kl_gauss = kl_gaussian(mu, log_var)

            kl_multi = kl_multinomial(gibbs)
            
            total_loss = reconst_loss  + kl_gauss + kl_multi
            if semi:
                log_likelihood = torch.log(g_softmax).squeeze(1)
                total_loss += 10*F.nll_loss(log_likelihood,labels,reduction='sum')
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            bs = images.size(0)
            bce += reconst_loss/bs
            kl += (kl_gauss + kl_multi)/bs
    
        denom = len(data_loader)
        nll = bce + kl
        if self.print_flag:
            print ("Train: NLL: {:.3f}, BCE:{:.3f}, KLGauss+KLMulti:{:.3f}".format(nll.item()/denom,
                                                                                   bce.item()/denom,
                                                                                   kl.item()/denom))
        return nll.item()/denom

    def evaluation(self,test_loader,text = 'Test'):
        with torch.no_grad():
            bce,kl,nll =0,0,0
            self.vae.eval()
            total = 0
            correct = 0
            for i, (images, labels) in enumerate(test_loader):
                images = to_var(images.view(images.size(0), -1))
                labels = to_var(labels)
                out,mu, log_var,gibbs,g_softmax = self.vae(images,self.tau,hard=True)
                reconst_loss = F.binary_cross_entropy_with_logits(out, images,reduction='sum')
                
                kl_gauss =  kl_gaussian(mu, log_var)
                kl_multi = kl_multinomial(gibbs)
                
                _, predicted = torch.max(g_softmax.squeeze(1), 1)
                total = labels.size(0)
                correct += (predicted == labels).sum().float()/total
                bs = images.size(0)
                bce += reconst_loss /bs
                kl += (kl_gauss + kl_multi)/bs
            nll = bce + kl
            denom = len(test_loader)
            if self.print_flag:
                print (text+": NLL: {:.3f}, BCE:{:.3f}, KLGauss+KLMulti:{:.3f}, accuracy: {:.3f}".format(nll.item()/denom,
                                                                                                        bce.item()/denom,
                                                                                                        kl.item()/denom,
                                                                                                        float(correct)/denom))
                                                                                                        
            self.vae.train()
            return nll.item()/denom,float(correct)/denom
    
    def training_procedure(self):
        
        torch.manual_seed(self.params['random_seed'])
        iter_per_epoch = len(self.train_loader)
        data_iter = iter(self.train_loader)
        
        ####### fixed inputs for debugging ######
        different_gaussians = 15
        bsize2print = different_gaussians*self.K
        ## for generating images from random numbers ##
        ind = torch.multinomial(torch.ones(bsize2print*self.N,self.K),1)
        ind = torch.zeros(bsize2print*self.N,1).long()
        for i in range(ind.size(0)):
            ind[i] = i%self.K
        self.fixed_z_d = to_var(torch.zeros(bsize2print*self.N, self.K).scatter_(1, ind, 1)).view(-1,self.N,self.K)
        fixed_z_c = duplicate_along(torch.randn(different_gaussians,self.gaussian_dim),self.K)
        self.fixed_z_c = to_var(duplicate_along(fixed_z_c,self.K))
        ## for reconstruction images ##
        fixed_x, _ = data_iter.next()
        path_to_save = './results/images/'
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        torchvision.utils.save_image(fixed_x.view(fixed_x.size(0),1, 28, 28),path_to_save+'real_images_gsm.png')
        self.fixed_x = to_var(fixed_x.view(fixed_x.size(0), -1))
        first_after_semi = True
        semi = False
        semi_epoch_list = list(range(self.epoch_semi_sup))
        train_nll,valid_nll,test_nll = [],[],[]
        valid_accuracy,test_accuracy = [],[]
        
        for epoch in range(1,self.num_epochs):
            self.print_flag = epoch % self.print_every == 0
            if self.print_flag:
                print (" ----- Epoch[{}/{}] ------".format(epoch, self.num_epochs))
            if epoch in semi_epoch_list:
               data_loader = self.train_loader_balanced
               semi = True
               #self.tau = 0.5
            else:
                data_loader = self.train_loader
                semi = False
                if first_after_semi:
                    self.tau= params['eps_0']
                    #params['ST-estimator'] = True
                    first_after_semi = False
                if self.print_flag and params['save_images']:
                    with torch.no_grad():
                        self.vae.eval()
                        reconst_images= F.sigmoid(self.vae.sample(self.fixed_z_d,self.fixed_z_c))
                        reconst_images = reconst_images.view(reconst_images.size(0), 1, 28, 28)
                        torchvision.utils.save_image(reconst_images.cpu(),
                                                     path_to_save+'generated_images_gsm_%d.png' %(epoch),nrow=self.K)
                        """
                        reconst_images,_, _,_,_= self.vae(fixed_x,self.tau,hard=True)
                        reconst_images = F.sigmoid(reconst_images)
                        reconst_images = reconst_images.view(reconst_images.size(0), 1, 28, 28)
                        torchvision.utils.save_image(reconst_images.cpu(),
                                                     './results/images/reconstructed_images_gsm_%d.png' %(epoch),nrow=self.K)
                        """
                        self.vae.train()
            nll = self.train(data_loader,semi)
            train_nll.append(nll)
            
            nll,acc = self.evaluation(self.valid_loader,'Validtion')
            valid_nll.append(nll)
            valid_accuracy.append(acc)
            
            nll,acc = self.evaluation(self.test_loader)
            test_nll.append(nll)
            test_accuracy.append(acc)
            if not semi:
                a = self.params['eps_0']*math.exp(-self.params['anneal_rate'] * epoch*len(self.train_loader))
                self.tau = np.maximum(a,self.params['min_eps']).item()

        nll_res = train_nll,valid_nll,test_nll
        acc_res = valid_accuracy,test_accuracy
        return nll_res,acc_res

"""
num_labels_data = [50,100,300,600]
all_results = []
for nld in num_labels_data:
    #params['supervised_epochs'] = int(1500/(nld*0.01))
    params['num_labeled_data'] = nld
    mixture_model = Mixture_Model(params)
    nll_res,acc_res = mixture_model.training_procedure()
    results = nll_res,acc_res,params.copy()
    all_results.append(results)
torch.save(all_results,'./results/semi_mixture_model_gsm_26sep.tar')

"""
