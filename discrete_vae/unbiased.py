
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import datas
from utils import *


# VAE model
class Encoder(nn.Module):
    def __init__(self, image_size=784, N=6,K=6,M=20):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(image_size,300),
            nn.ReLU(),
            nn.Linear(300,N*K))
        self.N = N
        self.K = K
        self.M = M     
    def forward(self, x):
        phi_x = self.encoder(x)
        return phi_x


class Decoder(nn.Module):
    def __init__(self, image_size=784,N=6,K=6):
        super(Decoder, self).__init__()
      
        self.decoder = nn.Sequential(
            nn.Linear(N*K, 300),
            nn.ReLU(),
            nn.Linear(300,image_size))
        
        self.N = N
        self.K = K

        
    def forward(self, y):
        out = self.decoder(y.view(-1,self.N*self.K))
        return out
    
class Direct_VAE:
    def __init__(self,params):

        self.N,self.K  = params['N_K'] 
        self.M = params['gumbels']
        self.encoder = Encoder(N=self.N,K=self.K)
        self.decoder = Decoder(N=self.N,K=self.K)
        self.eye = torch.eye(self.K)
        self.params = params

        print ('encoder: ',self.encoder)
        print ('decoder: ',self.decoder)

        if torch.cuda.is_available():
            self.encoder.cuda()
            self.decoder.cuda()
            
        lr = params['learning_rate']
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters())+list(self.decoder.parameters()), lr=lr)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.training_iter = 0


    def train(self,train_loader):    
        kl_sum,bce_sum = 0,0
        eps_0,ANNEAL_RATE,min_eps = self.params['eps_0'],self.params['anneal_rate'],self.params['min_eps']
        for i, (im, _) in enumerate(train_loader):
            images = to_var(im.view(im.size(0), -1))
            bs = im.size(0)
            ground_truth = images
            # forward
            phi_x = self.encoder(images)

            # backward
            thetas = self.enumerate_theta(phi_x,ground_truth)
            kl = kl_multinomial(phi_x)
            loss = torch.sum(F.softmax(phi_x,dim=1)*thetas)+ kl
            self.optimizer.zero_grad()

            loss.backward()
            
            self.optimizer.step()
   
            bce_sum += loss.detach()/bs


        nll_bce = (bce_sum.item())/len(train_loader)
        return nll_bce

    def evaluate(self,test_loader):
        self.encoder.eval()
        self.decoder.eval()

        bce_sum =0
        kl_div = 0
        with torch.no_grad():
            for images, _ in test_loader:        
                images = to_var(images.view(images.size(0), -1))
                ground_truth = images
                bs = images.size(0)
                phi_x = self.encoder(images)
                """
                hards = torch.zeros_like(phi_x).scatter_(-1, torch.argmax(phi_x,dim=1).view(-1,1),1.0)
                out = self.decoder(hards)
                decoder_loss  = self.bce_loss(out,ground_truth).sum()
                kl = kl_multinomial(phi_x)
                bce_sum += (decoder_loss+kl)/images.size(0)
                """
                thetas = self.enumerate_theta(phi_x,ground_truth)
                kl = kl_multinomial(phi_x)
                loss = torch.sum(F.softmax(phi_x,dim=1)*thetas)+ kl
                bce_sum += loss.detach()/images.size(0)
        self.encoder.train()
        self.decoder.train()

        nll_bce = bce_sum.item()/len(test_loader)
        return nll_bce

    def enumerate_theta(self,phi_x,ground_truth):
        
        N = self.N
        K = self.K
        assert N==1,'unbiased version works only for N==1'
        phi_x = phi_x.view(-1,N,K)
        self.decoder.eval()


        new_batch = []
        gt_batch = []

        for n in range(N):
            a_clone = torch.zeros_like(phi_x)
            idx = to_var(n*torch.ones(phi_x.size(0),1,phi_x.size(2)).long())
            a_clone.scatter_(1,idx, 0)

            for k in range(K):
                clone2 = a_clone.clone()
                clone2[:,n,k]=1
                new_batch.append(clone2)
                gt_batch.append(ground_truth)
    
        new_batch = torch.cat(new_batch,1)
        gt_batch = torch.cat(gt_batch,1).view(-1,ground_truth.size(-1))
        
        out = self.decoder(to_var(new_batch))
        losses = self.bce_loss(out,gt_batch).sum(dim = 1) #ground_truth.repeat(K*N,1)
        losses = losses.view(-1,K)
        return losses
      
def training_procedure(params):    
    """trains over the MNIST standard spilt (50K/10K/10K) or omniglot
    saves the best model on validation set
    evaluates over test set every epoch just for plots"""
    
    torch.manual_seed(params['random_seed'])

    train_loader,valid_loader,test_loader = datas.load_data(params)

    N,K = params['N_K']
    direct_vae = Direct_VAE(params)
    
    best_state_dicts = None
    print ('hyper parameters: ' ,params)

    train_results,valid_results,test_results = [],[],[]
    best_valid,best_test_nll = float('Inf'),float('Inf')

    for epoch in range(params['num_epochs']):
        epoch_results = [0,0,0]
        train_nll = direct_vae.train(train_loader)
        train_results.append(train_nll)
        epoch_results[0] = train_nll

        valid_nll  = direct_vae.evaluate(valid_loader) 
        valid_results.append(valid_nll)
        epoch_results[1] = valid_nll

        test_nll  = direct_vae.evaluate(test_loader) 
        test_results.append(test_nll)
        epoch_results[2] = test_nll
        
        if params['print_result']:        
            print_results(epoch_results,epoch,params['num_epochs'])        
       
        if valid_nll < best_valid:
            best_valid = valid_nll
            best_test_nll = test_nll
            best_state_dicts = (direct_vae.encoder.state_dict(),direct_vae.decoder.state_dict())

    return train_results,test_results,best_test_nll,best_state_dicts,params.copy()



