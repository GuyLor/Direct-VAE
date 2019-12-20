import torch
from torchvision import datasets
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, xrange(n, n-r, -1), 1)
    denom = reduce(op.mul, xrange(1, r+1), 1)
    return numer//denom

def kl_multinomial(logits_z):
    k = logits_z.size(-1)
    q_z = F.softmax(logits_z,dim =1)
    log_q_z = torch.log(q_z+1e-20)
    kl = q_z * (log_q_z - torch.tensor([1.0/k]).log().item())
    return kl.sum()

def get_data(batch_size,dataset='mnist'):
    ds = datasets.FashionMNIST if dataset=='fashion-mnist' else datasets.MNIST
    tr=torch.utils.data.DataLoader(
                    ds('./data/{}'.format(dataset),
                       train=True,
                       download=True,
                       transform=transforms.ToTensor()),
        
        batch_size=batch_size,
        shuffle=True)
    
    ts=torch.utils.data.DataLoader(
                    ds('./data/{}'.format(dataset),
                       train=False,
                       download=True,
                       transform=transforms.ToTensor()),
        
        batch_size=batch_size,
        shuffle=True)
    return tr,ts

def sample_gumbel(logits,beta=1.0,standard_gumbel=None, eps=1e-20):
    if standard_gumbel is None:
        g = torch.distributions.gumbel.Gumbel(logits,beta*torch.ones_like(logits))    
        return g.sample()
    else:
        U = torch.rand(logits.size()).float()
        return -beta*torch.log(eps - torch.log(U + eps)) 

def show_and_save_plots(lists, fig_name='new.png'):
        plt.clf()
        cls = ['orange','green','blue']
        leg=[]
        for i,l in enumerate(lists):
            c = cls[i]
            (train,test),label = l
            line1,=plt.plot(test,color=c,linestyle='-',label=label+' test')
            line2,=plt.plot(train,color=c,linestyle='--',label=label+' train')
            leg.append(line1)
            leg.append(line2)

        plt.ylabel('ELBO')
        plt.xlabel('Epochs')        
        plt.legend(handles=leg,loc='upper left', bbox_to_anchor=(1, 0.5))

        plt.show()

def show(img):
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

#######################################################################
#                              Structured                             #
#######################################################################

############ maxflow #################
import functools as ft
import maxflow
from itertools import combinations
import operator as op


def ncr(n, r):
    r = min(r, n-r)
    numer = ft.reduce(op.mul, range(n, n-r, -1), 1)
    denom = ft.reduce(op.mul, range(1, r+1), 1)
    return numer//denom

def get_argmax_and_max_maxflow(phi_i, phi_ij):
    # e.g.
    #n=3
    #phi_i = [phi_0, phi_1, phi_2]
    #phi_ij = [phi_{0,1}, phi_{0,2}, phi_{1,2}]
    phi_i, phi_ij = -phi_i, -phi_ij
    g = maxflow.Graph[float]()
    nids = g.add_nodes(len(phi_i))
    obj_idx = [i for i,_ in enumerate(phi_i)]

    mapping = {ij:ind for ind,ij in enumerate(combinations(range(len(phi_i)),2))}
    for i,j in list(mapping):
        mapping[(j,i)] = mapping[(i,j)]
    i =0
    while i < len(phi_i):
        j = i+1
        s = phi_i[i]
        p = 0.5
        t = sum(p*phi_ij[mapping[(i,j_)]] for j_ in range(len(phi_i)) if i != j_ ) + s

        g.add_tedge(i,t,0)

        while j < len(phi_i):
            c = phi_ij[mapping[(i,j)]]
            g.add_edge(i,j,-p*c,-p*c)
            j +=1
        i+=1    
    f = g.maxflow()
    return 1.0*g.get_grid_segments(nids),-f

def argmax_maxflow(h_list):
    k_batch = []
    for h_i,h_ij in h_list:
        h = h_i[:,1] - h_i[:,0]
        k,_ = get_argmax_and_max_maxflow(h.detach().cpu().numpy().astype(float),
                                       h_ij.reshape(-1).detach().cpu().numpy().astype(float))
        k_batch.append(k)
    k =torch.tensor(k_batch,dtype=torch.long)
    return k

def z_hat(z,h_g,N,K):
    h_g = h_g.view(-1,K)
    z = F.one_hot(z,K).float()
    z_copy = z.view(-1,N,K)
    new_batch = []
    for n in range(N):
        a_clone = z_copy.clone()
        idx = n*torch.ones(a_clone.size(0),1,a_clone.size(2)).long()
        a_clone.scatter_(1,idx, 0)

        for k in range(K):
            clone2 = a_clone.clone()
            clone2[:,n,k]=1
            new_batch.append(clone2)
    new_batch = torch.cat(new_batch,1).view(-1,N*2)
    return new_batch

def z_hat_bitwise(z):    
    new_batch = []
    N = z.size(-1)
    for n in range(N):
        a_clone = z.clone().bool()
        a_clone[:,n] = ~ a_clone[:,n]
        new_batch.append(a_clone.unsqueeze(1).float())

    new_batch = torch.cat(new_batch,1)
    return new_batch.view(-1,N)

def h_i_and_h_ij_grads(h_i,h_ij,f_theta,z_opt,epsilon=1.0,solver= 'maxflow'):
    K = h_i.size(-1)
    N = z_opt.size(-1)
    f_theta = epsilon*f_theta.view(-1,K)
    h_i = h_i - f_theta

    batch_h = zip(h_i.view(-1,N,K),h_ij)
    if solver == 'maxflow':
        argmax = argmax_maxflow(batch_h)
    elif solver=='cplex':
        argmax = argmax_cplex(batch_h)
    elif solver =='none':
        argmax = h_i.argmax(-1).view(-1,N)

    z_direct_1hot = F.one_hot(argmax,K)
    z_opt_1hot = F.one_hot(z_opt,K)
    
    z_direct = argmax.tolist()
    z_opt =  z_opt.tolist()

    
    
    gradients_sign_h_i = (z_opt_1hot - z_direct_1hot).view(-1,N,K) 
    grad_h_i = (gradients_sign_h_i*h_i.view(-1,N,K)).view(-1,N*K)   

    ij_sign_batch = []
    if solver =='none':
        h_ij = torch.cat(h_ij).view(grad_h_i.size(0),-1)
        ij_sign_batch = torch.zeros_like(h_ij)
    else:
        for zopt, zdirect in zip(z_opt,z_direct):
            zij_opt = torch.tensor([l*r for l,r in combinations(zopt,2)])
            zij_direct = torch.tensor([l*r for l,r in combinations(zdirect,2)])
            h_ij_sign_grad =  zij_opt - zij_direct
            ij_sign_batch.append(h_ij_sign_grad.unsqueeze(0))

        h_ij = [p.unsqueeze(0) for p in h_ij]
        h_ij = torch.cat(h_ij)
        ij_sign_batch =  torch.cat(ij_sign_batch).type(torch.float)
        
    grad_h_ij =ij_sign_batch*h_ij
    return grad_h_i,grad_h_ij



########### cplex ###############
import cplex

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
    k =torch.tensor(k_batch,dtype=torch.long)    
    return k



