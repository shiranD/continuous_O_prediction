import torch.nn as nn
import torch
from space import build_space
from scipy.special import ive
from scipy import signal
import pdb
import torch.distributions.normal as normal

class VAEloss(torch.nn.Module): # for gpu performance

    def __init__(self, dim=0):
        super().__init__()
        self.sampler = normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        self.cos = nn.CosineSimilarity(dim=dim)

    def samples(self, batch):
        sampling = self.sampler.sample([batch, 10]).cuda()
        return sampling.view(batch, 10)

    def compute_kernel(self, x, y):
        sigma = 2 / y.size()[0]
        return torch.exp(-(x - y).pow(2).mean(dim=1)) / sigma 
        
    def forward(self, z, recons, target):
        dim = z.size()[0]
        prior = self.samples(dim)
        prior_kernel = self.compute_kernel(prior, prior)
        train_kernel = self.compute_kernel(z,z)
        both_kernel = self.compute_kernel(prior, z)
        cos_loss = torch.sqrt(2*(1-self.cos(recons, target))).mean()
        return prior_kernel.mean() + train_kernel.mean() - 2 * both_kernel.mean() + cos_loss 

class MMDloss(torch.nn.Module): # for gpu performance

    def __init__(self):
        super().__init__()
        self.sampler = normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

    def samples(self, batch):
        sampling = self.sampler.sample([batch, 10]).cuda()
        return sampling.view(batch, 10)

    def compute_kernel(self, x, y):
        sigma = 2 / y.size()[0]
        return torch.exp(-(x - y).pow(2).mean(dim=1)) / sigma 
        
    def forward(self, z):
        dim = z.size()[0]
        prior = self.samples(dim)
        prior_kernel = self.compute_kernel(prior, prior)
        train_kernel = self.compute_kernel(z,z)
        both_kernel = self.compute_kernel(prior, z)
        return prior_kernel.mean() + train_kernel.mean() - 2 * both_kernel.mean() 

class VAELoss_sqrt(torch.nn.Module): # for gpu performance

    def __init__(self, num_samples, dim=0):
        super().__init__()
        self.num_samples = num_samples
        self.cos = nn.CosineSimilarity(dim=dim)

    def forward(self, input, target, mu, logvar):
        KLD_loss = torch.mean(-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())) #input size
        cos_loss = torch.sqrt(2*(1-self.cos(input, target))).mean()
        #pdb.set_trace()
        return cos_loss + KLD_loss*10*self.num_samples/len(target)


def Bessel(m,k):
    return torch.from_numpy(ive([m], k.detach().numpy())).float()

class NLLvMF(torch.nn.Module):

    def __init__(self, emdim, dim=0):
        super().__init__()
        self.emdim = float(emdim)
        self.lambda1 = 1
        self.lambda2 = 1
        self.cos = nn.CosineSimilarity(dim=dim)

  
    class logCmk(torch.autograd.Function):

        @staticmethod
        def forward(ctx, k):
           ctx.save_for_backward(k)
           two_pi = torch.tensor(2*3.141592653589793)
           m = 50
           #two_pi = torch.tensor(2*3.141592653589793).cuda()
           answer = (m/2-1)*torch.log(k) - torch.log(Bessel(m/2-1, k)) - k - (m/2)*torch.log(two_pi)
           return answer

        @staticmethod
        def backward(ctx, grad_output): # Bessel grad
           k, = ctx.saved_tensors
           m = 50
           x = -Bessel(m/2, k)/Bessel(m/2-1,k)
           #print(grad_output * torch.autograd.Variable(x) ,grad_output, torch.autograd.Variable(x))
           return grad_output * torch.autograd.Variable(x)

    #def cos_distance(self, preds, target):
    #    return torch.sqrt(2*(1-self.cos(preds, target)))

    def forward(self, input, target):
        #pdb.set_trace()
        kappa = input.norm(p=1, dim=1)
        logCmk = self.logCmk.apply
        #coss_loss = self.cosine_distance(input,target).sum()        
        #pdb.set_trace()
        #nll_loss = - self.lambda2 * self.cos(input,target)
        nll_loss = - logCmk(kappa)
        #nll_loss = - logCmk(kappa) - self.lambda2 * self.cos(input,target)
        #nll_loss = - logCmk(kappa) + torch.log(1+kappa) * (self.lambda1 - self.cos(input,target))
        #nll_loss.backward()
        #grad_output = None if input.grad is None else input.grad.data 
        #torch.all(nll_loss.grad, target)
        #print('before',nll_loss)
        nll_loss = nll_loss.mean()
        print('after',nll_loss)
        return nll_loss
    
class MaxMarginLoss(torch.nn.Module): # for gpu performance

    def __init__(self,emdim, veclist, dim=0):
        super().__init__()
        #self.gamma = torch.tensor(0.5)
        self.gamma = torch.tensor(0.5).cuda()
        self.rank = 2 # one extra in case the target is there
        self.emdim = emdim
        #self.veclist = torch.tensor(veclist).float()
        self.veclist = torch.tensor(veclist).cuda().float() # cuda
        self.cos = nn.CosineSimilarity(dim=dim)
        self.space = build_space(emdim, veclist)
        self.space.build(100)
        self.gamma_vec = self.gamma.repeat(self.rank) 
    
    def cos_distance(self, preds, target):
        return torch.sqrt(2*(1-self.cos(preds, target)))
        
    def forward(self, input, target):

        cost = 0
        batch, _ = input.shape
        tg_dists = self.cos_distance(input,target)

        for pred, trg, dist in zip(input, target, tg_dists):

            # instead of sampling cosider the prediction's vicinity
            nearest = self.space.get_nns_by_vector(pred, n=self.rank, search_k=-1)

            #vecs = torch.empty(self.rank-1, self.emdim).float()
            vecs = torch.empty(self.rank-1, self.emdim).cuda().float()
            i=0
            for friend in nearest:
                if i == self.rank-1: # only top rank friends
                    break
                vec = self.veclist[friend]
                if torch.all(torch.eq(vec, trg)): # ignore target
                    continue
                vecs[i,:] = vec
                i+=1
            
            sample_dist = self.gamma_vec + dist.repeat(self.rank-1) - self.cos_distance(pred.repeat(self.rank-1,1), vecs)
            # sum all positive values
            sample_cost = torch.sum(sample_dist[sample_dist>0])

            cost+=sample_cost
        #print(cost/batch, dist, self.cos_distance(pred.repeat(self.rank-1,1), vecs) )
        return cost / batch # mean

                
class SIMLoss_sqrt(torch.nn.Module): # for gpu performance

    def __init__(self, dim=0):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=dim)

    def forward(self, input, target):
        return torch.sqrt(2*(1-self.cos(input, target))).mean()


class GPLoss(torch.nn.Module): # for gpu performance

    def __init__(self, dim=0):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=dim)
        self.alpha = 0.7

    def forward(self, input, target):
        return torch.sqrt(1-self.cos(self.alpha * input + (1 - self.alpha) * target, target)).mean()

class SIMLoss(torch.nn.Module): # for gpu performance

    def __init__(self, dim=0):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=dim)

    def forward(self, input, target):
        return 1-self.cos(input, target).mean()


class Log_loss(torch.nn.Module):
    def __init__(self):
        # negation is true when you minimize -log(val)
        super(Log_loss, self).__init__()
       
    def forward(self, x, negation=True):
        # shape of x will be [batch size]
        log_val = torch.log(x)
        loss = torch.sum(log_val)
        if negation:
            loss = torch.neg(loss)
        return loss
    
class Itself_loss(torch.nn.Module):
    def __init__(self):
        super(Itself_loss, self).__init__()
        
    def forward(self, x, negation=True):
        # shape of x will be [batch size]
        loss = torch.sum(x)
        if negation:
            loss = torch.neg(loss)
        return loss
