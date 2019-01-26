
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
import torchvision
import numpy as np
import datas
from utils import *

# VAE model

class VAE(nn.Module):
    def __init__(self, image_size=784, N=6, K=6,M=20,tau=1.0, st_estimator=False,composed_decoder=True):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(image_size,300),
            nn.ReLU(),
            nn.Linear(300,N*K))
        
        self.composed_decoder = composed_decoder
        if not composed_decoder:
            linear_combines = [nn.Sequential(nn.Linear(K, 300),
                                             nn.ReLU(),
                                             nn.Linear(300,image_size)) for _ in range(N)]
            self.decoder = ListModule(*linear_combines)
        else:
            self.decoder = nn.Sequential(
                nn.Linear(N*K, 300),
                nn.ReLU(),
                nn.Linear(300,image_size))

        self.K = K
        self.N = N
        self.M = M
        self.tau = tau
        self.st = st_estimator
        
                     
    def forward(self, x):        
        logits = self.encoder(x).view(-1,self.K)
        logits1 = logits.repeat(self.M,1)       

        if self.st:
            y = self.gumbel_softmax(logits1, self.tau, hard = True) # if training: forward and backward with relaxed
        else:
            y = self.gumbel_softmax(logits1, self.tau, hard = not self.training)
            
        if not self.composed_decoder:
            y=y.view(-1,self.N,self.K)
            bs = y.size(0)
            n_out = []
            for n in range(self.N):
                nth_input = y[:,n,:].view(-1,self.K)
                n_out.append(F.sigmoid(self.decoder[n](nth_input)))

            out = torch.stack(n_out,1) .mean(1)
        else:
            out = self.decoder(y.view(-1,self.N*self.K))        
        
      
        return out, logits
    
    def sample(self, z):
        return self.decoder(z)
    
    def sample_gumbel(self, shape, eps=1e-20):
            #Sample from Gumbel(0, 1)
            U = torch.rand(shape).float()
            return - torch.log(eps - torch.log(U + eps))

    def gumbel_softmax_sample(self,logits, tau=1, eps=1e-20):        
        gumbel_noise = self.sample_gumbel(logits.size())
        y = logits + to_var(gumbel_noise)
        return F.softmax(y / tau,dim=1)

    def gumbel_softmax(self,logits, tau=1, hard=False, eps=1e-10):

        y_soft = self.gumbel_softmax_sample(logits, tau=tau, eps=eps)

        if hard:
            _, k = y_soft.data.max(-1)
            shape = logits.size()
            y_hard = torch.FloatTensor(*shape).zero_().scatter_(-1, k.view(-1, 1), 1.0)
            y_hard = to_var(y_hard)
            # trick for forward one-hot and backward wrt soft
            y = (y_hard - y_soft).detach() + y_soft
        else:
            y = y_soft
        return y

class GSM_VAE:
    def __init__(self,params):
        self.N,self.K  = params['N_K']
        self.annealing_rate = params['anneal_rate']        
        self.vae = VAE(image_size=784,
                       N = self.N,
                       K = self.K,
                       M =params['gumbels'],
                       tau = params['eps_0'],
                       st_estimator=params['ST-estimator'],
                       composed_decoder = params['composed_decoder'])
        self.params = params        
        
        print ('vae model: ',self.vae)
        
        if torch.cuda.is_available():
            self.vae.cuda()

        lr = params['learning_rate']
        self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)
        self.training_iter = 0
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)

    def train(self,data_loader):
        nll_sum = 0
        eps_0,ANNEAL_RATE,min_eps = self.params['eps_0'],self.params['anneal_rate'],self.params['min_eps']
        #scheduler.step()
        for i, (im, _) in enumerate(data_loader):        
            images = to_var(im.view(im.size(0), -1))
            out, logits = self.vae(images)
            bs = images.size(0)

            KL = kl_multinomial(logits)
            reconst_loss = F.binary_cross_entropy_with_logits(out,images,size_average=False)
            total_loss = (reconst_loss + KL)/bs
            self.vae.zero_grad()
            
            total_loss.backward()
            
            self.optimizer.step()
            if self.training_iter % 1000 == 0:
                a = eps_0*math.exp(-ANNEAL_RATE*self.training_iter)
                self.vae.tau = np.maximum(a,min_eps)

            nll_sum += total_loss.detach().item()
            self.training_iter += 1

        nll_mean = nll_sum/len(data_loader)
        return nll_mean

    def evaluate(self,test_loader):
        with torch.no_grad():
            nll_sum = 0
            self.vae.eval()
            self.vae.M = 100
            for i, (im, _) in enumerate(test_loader):
                images = to_var(im.view(im.size(0), -1))
                bs = im.size(0)
                out, logits = self.vae(images)

                labels = images.repeat(self.vae.M,1)
                KL = kl_multinomial(logits)
                reconst_loss = F.binary_cross_entropy_with_logits(out,labels,reduce = False)
                reconst_loss = reconst_loss.view(self.vae.M,bs,-1).mean(0).sum()
                bce_gumbel = (reconst_loss+KL)/bs
                nll_sum += bce_gumbel.item()

            nll_mean = nll_sum/len(test_loader)
            self.vae.M = self.params['gumbels']
            self.vae.train()
        return nll_mean

def training_procedure(params):
    torch.manual_seed(params['random_seed'])

    train_loader,valid_loader,test_loader = datas.load_data(params)
    N,K = params['N_K']
    num_epochs = params['num_epochs']
    gsm_vae = GSM_VAE(params)
    best_state_dicts = None
    print ('hyper parameters: ' ,params)

    train_results,valid_results,test_results = [],[],[]
    best_valid,best_test_nll = float('Inf'),float('Inf')

    for epoch in range(num_epochs):
        epoch_results = [0,0,0] 
        train_nll = gsm_vae.train(train_loader)
        train_results.append(train_nll)
        epoch_results[0] = train_nll

        valid_nll  = gsm_vae.evaluate(valid_loader) 
        valid_results.append(valid_nll)
        epoch_results[1] = valid_nll

        test_nll  = gsm_vae.evaluate(test_loader) 
        test_results.append(test_nll)
        epoch_results[2] = test_nll
        
        if params['print_result']:        
            print_results(epoch_results,epoch,params['num_epochs'])         
        
        if valid_nll < best_valid:
            best_valid = valid_nll
            best_test_nll = test_nll
            best_state_dict = gsm_vae.vae.state_dict()

    return train_results,test_results,best_test_nll,best_state_dict,params.copy()
