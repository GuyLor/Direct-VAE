import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import operator as op
from utils import *
import datas
import cplex
from itertools import combinations
from itertools import product
import time

params = {'num_epochs': 300,
            'batch_size': 100,
            'learning_rate': 0.001,
            'gumbels' : 1,
            'N_K': (15,2),
            'eps_0':1.0,
            'anneal_rate':1e-5,
            'min_eps':0.1,
            'structure': False,
            'dataset':'mnist',
            'split_valid':False,
            'binarize':True,
            'random_seed':777,
            'print_model':True,
            'print_result':True}


is_cuda = torch.cuda.is_available()
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer//denom
def setproblemdata(p, phi_i, phi_ij):
    #n=3
    #phi_i = [phi_0, phi_1, phi_2]
    #phi_ij = [phi_{0,1}, phi_{0,2}, phi_{1,2}]
    p.set_problem_name ("example")
    p.objective.set_sense(p.objective.sense.maximize)
    ub   = [1.0 for _ in phi_i]
    lb   = [0.0 for _ in phi_i]
    var_type = 'I'*len(phi_i)
    obj_idx = [i for i,_ in enumerate(phi_i)]
    p.variables.add(obj = phi_i, ub = ub, lb = lb, types=var_type) #, names = ["z0", "z1", "z2"])

    mapping = {ij:ind for ind,ij in enumerate(combinations(range(len(phi_i)),2))}
    for i,j in list(mapping):
        mapping[(j,i)] = mapping[(i,j)]
    qmat = []
    for i,_ in enumerate(phi_i):
        row = [obj_idx]
        q = []
        for j,_ in enumerate(phi_i):
            if i==j:
                q.append(0.0)
            else:
                q.append(phi_ij[mapping[(i,j)]])
        row.append(q)
        qmat.append(row)
    p.objective.set_quadratic(qmat)
    
def get_argmax_and_max(phi_i, phi_ij):
    p = cplex.Cplex()

    p.set_log_stream(None)
    p.set_error_stream(None)
    p.set_warning_stream(None)
    p.set_results_stream(None)

    assert len(phi_ij) == len(list(combinations(range(len(phi_i)),2)))

    setproblemdata(p, phi_i, phi_ij)
    p.solve()
    sol = p.solution
    z = list(map(sol.get_values, range(len(phi_i))))
    sol_val = sol.get_objective_value()
    return z,sol_val

def argmax_cplex(phi_list):
    
    k_batch = []
    for phi_i,phi_ij in phi_list:

        phi = phi_i[:,1] - phi_i[:,0]
        k,_ = get_argmax_and_max(phi.detach().cpu().numpy().astype(float),phi_ij.reshape(-1).detach().cpu().numpy().astype(float))
        k_batch.append(k)

    k =to_var(torch.tensor(k_batch,dtype=torch.long))
    #print (k.size())
    return k


    #print 
class Encoder(nn.Module):
    def __init__(self, image_size=784, N=6,K=6,M=20):
        super(Encoder, self).__init__()
        self.comb_num = ncr(N,K)
        self.encoder = nn.Sequential(
            nn.Linear(image_size,300),
            nn.ReLU(),
            nn.Linear(300,N*K+self.comb_num))
        
        #self.encode_phi_ij = nn.Sequential(
        #    nn.Linear(image_size,self.comb_num))

        self.N = N
        self.K = K
        self.M = M
        
        #print self.comb_num
    def forward(self, x,structure=True):
        phi_x = self.encoder(x)
        #phi_i_j = self.encode_phi_ij(x)
        phi_i, phi_i_j = phi_x.split([self.N*self.K,self.comb_num],dim=1)
        
        z,phi_x_g = self.gumbel_perturbation(phi_i,phi_i_j,structure)
        return z,phi_x_g,phi_x 

    def sample_gumbel(self,shape, eps=1e-20):
        #Sample from Gumbel(0, 1)
        U = torch.rand(shape).float()
        return -torch.log(eps - torch.log(U + eps))
    
    def gumbel_perturbation(self,phi_x,phi_i_j,structure = True, eps=1e-10):
        M,K,N = self.M,self.K,self.N
        phi_x=phi_x.contiguous().view(-1,K)
        phi_x = phi_x.repeat(M,1)
        shape = phi_x.size()
        gumbel_noise = to_var(self.sample_gumbel(shape, eps=eps))
        phi_x_gamma = phi_x + gumbel_noise
        batch_phi =list( zip(phi_x_gamma.view(-1,N,K),phi_i_j))
        #global self.structure
        if structure:
            k = argmax_cplex(batch_phi)
        else:
            _, k = phi_x_gamma.data.max(-1)
        if is_cuda:
            z = torch.cuda.FloatTensor(*shape).zero_().scatter_(-1, k.view(-1, 1), 1.0)
        else:
            z = torch.FloatTensor(*shape).zero_().scatter_(-1, k.view(-1, 1), 1.0)
        z_phi_gamma = to_var(z)
        return z_phi_gamma,(batch_phi,k) #phi_x_gamma
        

class Decoder(nn.Module):
    def __init__(self, image_size=784,N=6,K=6,M=3):
        super(Decoder, self).__init__()       

        self.decoder = nn.Sequential(
            nn.Linear(N*K, 300),
            nn.ReLU(),
            nn.Linear(300,image_size))
        self.N = N
        self.K = K
        self.M = M
        
    def forward(self, y):
        out = self.decoder(y.view(-1,self.N*self.K))
        return out
    
class Direct_VAE:
    def __init__(self,params):

        self.N,self.K  = params['N_K'] 
        self.M = params['gumbels']
        self.encoder = Encoder(N=self.N,K=self.K,M=self.M)
        self.decoder = Decoder(N=self.N,K=self.K,M=self.M)
        self.eps = params['eps_0']
        self.annealing_rate = params['anneal_rate']
        self.structure = params['structure']
        self.params = params
        if params['print_model']:
            
            params['print_model'] = False
            print ('encoder: ',self.encoder)
            print ('decoder: ',self.decoder)

        if torch.cuda.is_available():
            self.encoder.cuda()
            self.decoder.cuda()
            
        lr = params['learning_rate']
        self.optimizer_e = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.optimizer_d = torch.optim.Adam(self.decoder.parameters(), lr=lr)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction = 'none') #size_average=False, reduce=False
        self.training_iter = 0

    def train(self,train_loader,return_time=None):    
        kl_sum,bce_sum = 0,0
        eps_0,ANNEAL_RATE,min_eps = params['eps_0'],params['anneal_rate'],params['min_eps'] 
        training_time = []
        for i, (im, _) in enumerate(train_loader):
            start = time.time()
            images = to_var(im.view(im.size(0), -1))
            bs = im.size(0)
            ground_truth = images.repeat(self.M,1)
            # forward
            z_hard,phi_x_g,phi_x = self.encoder(images,structure=self.structure)
            out = self.decoder(z_hard)

            # backward

            encoder_loss = self.compute_encoder_gradients(z_hard,phi_x_g,ground_truth,self.eps)

            if self.M !=0:
                decoder_loss  = self.bce_loss(out,ground_truth).view(self.M,bs,-1).mean(0).sum()
            else:
                decoder_loss  = self.bce_loss(out,ground_truth).sum()
            
            kl = kl_multinomial(phi_x[:,:self.N*self.K].contiguous().view(-1,self.K))

            #encoder_loss += kl
            self.optimizer_e.zero_grad()
            self.optimizer_d.zero_grad()

            decoder_loss.backward()
            encoder_loss.backward()

            self.optimizer_d.step()
            self.optimizer_e.step()
            
            bce_sum += (decoder_loss+kl)/bs
            if self.training_iter % 500 == 0:
                a = eps_0*math.exp(-ANNEAL_RATE*self.training_iter)
                if a > min_eps:
                    self.eps = a
                else:
                    self.eps=min_eps
            
            self.training_iter += 1
            end = time.time()
            training_time.append(end-start)

        nll_bce = (bce_sum.item())/len(train_loader)
        avg_time = np.mean(training_time)
        to_return =nll_bce
        
        if return_time:
            to_return = avg_time
        return to_return

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
                hards,_,phi_x = self.encoder(images,structure=self.structure)

                out = self.decoder(hards)

                #print self.bce_loss(out,ground_truth).sum().item()
                decoder_loss  = self.bce_loss(out,ground_truth).view(self.M,bs,-1).mean(0).sum()
                kl = kl_multinomial(phi_x[:,:self.N*self.K].contiguous().view(-1,self.K))

                bce_sum += (decoder_loss+kl)/images.size(0)
    

        self.encoder.train()
        self.decoder.train()
        self.encoder.M = params['gumbels']
        self.decoder.M = params['gumbels']
        self.M = params['gumbels']
        nll_bce = bce_sum.item()/len(test_loader)
        return nll_bce
    def compute_encoder_gradients(self,z_hard,phi_x_g,ground_truth,epsilon=1.0):
        N = self.N
        K = self.K
        phi_x_g,z_opt = phi_x_g
        #phi_x_g = list(phi_x_g)
        soft_copy,phi_ij = zip(*phi_x_g)
        soft_copy = torch.cat(soft_copy)
        soft_copy = soft_copy.view(-1,K)
        hard_copy = z_hard.data.view(-1,N,K)
        self.decoder.eval()
        new_batch = []
        gt_batch = []
        for n in range(N):
            a_clone = hard_copy.clone()
            idx = n*torch.ones(hard_copy.size(0),1,hard_copy.size(2)).long()
            if is_cuda:
                idx = idx.cuda()
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
        if not self.structure:
            _, k = soft_copy.max(-1)
        else:
            batch_phi = zip(soft_copy.view(-1,N,K),phi_ij)
            k = argmax_cplex(batch_phi)#.view(-1,K)
        z_direct = k.tolist()
        z_opt =  z_opt.tolist()

        if is_cuda:
            change = torch.cuda.FloatTensor(*shape).zero_().scatter_(-1, k.view(-1, 1),1.0)
        else:
            change = torch.FloatTensor(*shape).zero_().scatter_(-1, k.view(-1, 1),1.0)
        gradients_sign_phi_i = (hard_copy - change).view(-1,N,K)
        phi_i = soft_copy.view(-1,N,K)
        grad_phi_i = (gradients_sign_phi_i*phi_i).view(-1,N*K)
        ij_sign_batch = []
        if self.structure:
            for zopt, zdirect in zip(z_opt,z_direct):
                zij_opt = torch.tensor([l*r for l,r in combinations(zopt,2)])
                zij_direct = torch.tensor([l*r for l,r in combinations(zdirect,2)])
                phi_ij_sign_grad =  zij_opt - zij_direct
                ij_sign_batch.append(phi_ij_sign_grad.unsqueeze(0))

            phi_ij = [p.unsqueeze(0) for p in phi_ij]
            phi_ij = torch.cat(phi_ij)
            ij_sign_batch = to_var(torch.cat(ij_sign_batch).type(torch.float))
        else:
            phi_ij = torch.cat(phi_ij).view(grad_phi_i.size(0),-1)
            ij_sign_batch = torch.zeros_like(phi_ij)
        grad_phi_ij = ij_sign_batch*phi_ij
        gradients = torch.cat((grad_phi_i, grad_phi_ij),-1)
        self.decoder.train()
        gradients = gradients*(1.0/epsilon)
        return torch.sum(gradients)

def training_procedure(params):    
    """Trains over the MNIST standard spilt (50K/10K/10K) 
    Saves the best model on validation set
    Evaluates over test set every epoch for plots"""

    train_loader,test_loader = datas.load_data(params)
    
    N,K = params['N_K']
    direct_vae = Direct_VAE(params)
    
    best_state_dicts = None
    print( 'hyper parameters: ' ,params)

    train_results,test_results = [],[]

    print ('len(train_loader)', len(train_loader))
    print ('len(test_loader)', len(test_loader))
    for epoch in range(params['num_epochs']):
        epoch_results = [0,0]
        train_nll = direct_vae.train(train_loader)
        train_results.append(train_nll)
        epoch_results[0] = train_nll

        test_nll  = direct_vae.evaluate(test_loader) 
        test_results.append(test_nll)

        epoch_results[1] = test_nll
        if params['print_result']:        
            print_results(epoch_results,epoch,params['num_epochs'])

    return train_results,test_results

def check_running_time(params):
    torch.manual_seed(params['random_seed'])
    params['batch_size']=1
    train_loader,test_loader = datas.load_data(params)
    print ('len(train_loader)', len(train_loader))
    print ('len(test_loader)', len(test_loader))  
    print ('hyper parameters: ' ,params)
    time_to_plot = []
    nk = [(i,2) for i in range(5,16)]
    
    for n_k in nk:
        params['N_K']=n_k
        direct_vae = Direct_VAE(params)
        time = direct_vae.train(train_loader,return_time=True)
        time_to_plot.append((n_k[0],time))
        print (n_k,time)
    return time_to_plot 
#nk = [(1,64),(2,32),(1,64),(1,64)]
results = training_procedure(params)



