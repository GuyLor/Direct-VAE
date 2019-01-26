
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
        phi_x = self.encoder(x).view(-1,self.K)
        z,phi_x_g = self.gumbel_perturbation(phi_x)
        return z,phi_x_g,phi_x 

    def sample_gumbel(self,shape, eps=1e-20):
        #Sample from Gumbel(0, 1)
        U = torch.rand(shape).float()
        return -torch.log(eps - torch.log(U + eps))
    
    def gumbel_perturbation(self,phi_x, eps=1e-10):
        M,K,N = self.M,self.K,self.N

        phi_x = phi_x.repeat(M,1)
        shape = phi_x.size()
        gumbel_noise = to_var(self.sample_gumbel(shape, eps=eps))
        phi_x_gamma = phi_x + gumbel_noise
        # hard:
        _, k = phi_x_gamma.data.max(-1)

        z_phi_gamma = to_var(torch.FloatTensor(*shape)).zero_().scatter_(-1, k.view(-1, 1), 1.0)

        return z_phi_gamma,phi_x_gamma

class Decoder(nn.Module):
    def __init__(self, image_size=784,N=6,K=6,composed_decoder = True):
        super(Decoder, self).__init__()
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
        
        self.N = N
        self.K = K

        
    def forward(self, y):

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
        return out
    
class Direct_VAE:
    def __init__(self,params):

        self.N,self.K  = params['N_K'] 
        self.M = params['gumbels']
        self.encoder = Encoder(N=self.N,K=self.K,M=self.M)
        self.decoder = Decoder(N=self.N,K=self.K, composed_decoder=params['composed_decoder'])
        self.eps = params['eps_0']
        self.annealing_rate = params['anneal_rate']

        self.params = params

        print ('encoder: ',self.encoder)
        print ('decoder: ',self.decoder)

        if torch.cuda.is_available():
            self.encoder.cuda()
            self.decoder.cuda()
            
        lr = params['learning_rate']
        self.optimizer_e = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.optimizer_d = torch.optim.Adam(self.decoder.parameters(), lr=lr)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.training_iter = 0


    def train(self,train_loader):    
        kl_sum,bce_sum = 0,0
        eps_0,ANNEAL_RATE,min_eps = self.params['eps_0'],self.params['anneal_rate'],self.params['min_eps']
        for i, (im, _) in enumerate(train_loader):
            images = to_var(im.view(im.size(0), -1))
            bs = im.size(0)
            ground_truth = images.repeat(self.M,1)
            # forward
            z_hard,phi_x_g,phi_x = self.encoder(images)
            out = self.decoder(z_hard)

            # backward
            gradients = self.compute_encoder_gradients(z_hard,phi_x_g,ground_truth,self.eps)
            encoder_loss = torch.sum(to_var(gradients)*phi_x_g) 
            
            decoder_loss  = self.bce_loss(out,ground_truth).view(self.M,bs,-1).mean(0).sum()
            kl = kl_multinomial(phi_x)
            
            #decoder_loss += kl
            self.optimizer_e.zero_grad()
            self.optimizer_d.zero_grad()

            encoder_loss.backward()
            decoder_loss.backward()
            
            
            self.optimizer_d.step()
            self.optimizer_e.step()
   
            bce_sum += (decoder_loss+kl).detach()/bs
            
            if self.training_iter % 500 == 0:
                a = eps_0*math.exp(-ANNEAL_RATE*self.training_iter)
                self.eps = np.maximum(a,min_eps).item()
            self.training_iter += 1

        nll_bce = (bce_sum.item())/len(train_loader)
        return nll_bce

    def evaluate(self,test_loader):
        self.encoder.eval()
        self.decoder.eval()
        #self.encoder.M = 100
        #self.decoder.M = 100
        #self.M = 100
        bce_sum =0
        kl_div = 0
        with torch.no_grad():
            for images, _ in test_loader:        
                images = to_var(images.view(images.size(0), -1))
                ground_truth = images.repeat(self.M,1)
                bs = images.size(0)
                hards,_,phi_x = self.encoder(images)
                out = self.decoder(hards)

                decoder_loss  = self.bce_loss(out,ground_truth).view(self.M,bs,-1).mean(0).sum()
                kl = kl_multinomial(phi_x)
                bce_sum += (decoder_loss+kl)/images.size(0)

        self.encoder.train()
        self.decoder.train()
        self.encoder.M = self.params['gumbels']
        self.decoder.M = self.params['gumbels']
        self.M = self.params['gumbels']
        nll_bce = bce_sum.item()/len(test_loader)
        return nll_bce

    def compute_encoder_gradients(self,z_hard,phi_x_g,ground_truth,epsilon=1.0):
        with torch.no_grad():
            N = self.N
            K = self.K
            soft_copy = phi_x_g.data
            hard_copy = z_hard.data.view(-1,N,K)
            self.decoder.eval()

            new_batch = []
            gt_batch = []
            for n in range(N):
                a_clone = hard_copy.clone()
                idx = to_var(n*torch.ones(hard_copy.size(0),1,hard_copy.size(2)).long())
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
            hard_copy = hard_copy.view(-1,K)
            losses = epsilon*losses.view(-1,K).data
            soft_copy = soft_copy - losses
            shape = soft_copy.size()
            _, k = soft_copy.max(-1)
              
            change = to_var(torch.FloatTensor(*shape).zero_()).scatter_(-1, k.view(-1, 1),1.0)
            gradients = hard_copy - change
            self.decoder.train()
            gradients = gradients*(1.0/epsilon)
        return gradients
      
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



