import random
import numpy as np
from torch.autograd import Variable
import torch
import copy
import pdb
import sys

random.seed(9001)

class Samples_Generator:

    def __init__(self, train_dict, sample, dim, veclist, cuda, pos=False, path=None, mode='eval'):
        
        self.vec = torch.tensor(train_dict["ids"]) # data
        self.keep = train_dict["len2sentidx"]
        self.copysent = {}
        self.sample_size = sample
        self.dim = dim
        self.cuda = cuda
        self.history_len = 1
        self.const_window = 50
        self.total_sent = train_dict["total"]
        self.pos = pos
        self.path = path
        self.make_emb(veclist)
        self.train_size = 1
        self.nsamples = min(60000, self.total_sent) # or a function of keep
        #self.nsamples = 5 # or a function of keep
        if self.cuda:
            self.vec = self.vec.cuda()
        if self.pos and mode == 'eval':
            self.make_pos(self.path)
        if mode == 'eval':
            self.sorted_idx()  
        if self.pos and mode == 'test':
            self.make_pos_d(self.path)
 
    def sorted_idx(self):
        """
        Sort senteces idx
        """
        idxs = []
        for _, val in self.keep.items():
            idxs.extend(val)
        idxs = sorted(idxs)
        idxs.append(len(self.vec)) # get total length
        self.idxs = idxs 

    def make_pos(self, path):
        """
        Serialize POS
        """
        pos = []
        for line in open(self.path+'_pos', 'r').readlines():
            sent = ['<S>']
            line = line.strip()
            line = line.split()
            sent+=line+['</S>']
            pos.extend(sent)
        self.pos_idx = pos

    def make_pos_d(self, path):
        """
        Serialize POS
        """
        pos = []
        pos2int = {}
        int2pos = {}
        m=0
        for line in open(self.path+'_pos', 'r').readlines():
            sent = ['<S>']
            line = line.strip()
            line = line.split()
            sent+=line+['</S>']
            for p in sent:
                if p not in pos2int:
                    pos2int[p]=m
                    int2pos[m]=p
                    m+=1
                pos.append(pos2int[p])
        print(len(pos), self.vec.shape)
        self.pos2int = pos2int
        self.int2pos = int2pos
        pos = torch.tensor(pos)
        if self.cuda:
            pos = pos.cuda()
        self.pos_idx = pos

    def make_emb(self, veclist):
        """
        Make an emb layer
        """
        embed = torch.nn.Embedding(len(veclist), len(veclist[0]))
        embed.weight.data.copy_(torch.from_numpy(np.array(veclist)))
        self.veclist = embed
        if self.cuda:
            self.veclist = self.veclist.cuda()

    def choose_len(self):
        """
        choose history length to train on
        """
        if not bool(self.copysent):
            print("a new round of {} train sentences".format(self.total_sent))
            self.copysent = copy.deepcopy(self.keep)
            self.new = True
        self.history_len = random.sample(list(self.copysent.keys()), 1)[0] 
            
    def number_of_sent(self):
        """
        Retrive how many sentences are found in the required length
        """
        self.number_of_sents = len(self.copysent[self.history_len])

    def final_size(self):
        """
        Compute the final sample size possible
        """
        self.train_size = min(self.number_of_sents, self.sample_size)

    def choose_window(self):
        """
        Compute the final sample size possible
        """
        self.window = min(self.history_len, self.const_window)

    def subTensor(self):
        """
        Extract the samples from the idx flat vector
        """
        sub = Variable(
            torch.zeros(
                self.history_len + 1,
                self.train_size, dtype=torch.long),
            requires_grad=False)

        values = copy.deepcopy(self.copysent[self.history_len])
        if self.pos:
            self.val4pos = copy.deepcopy(values)
        for y, idx in enumerate(self.copysent[self.history_len]):
            sub[:,y] = self.vec[idx:idx+self.history_len+1]
            values.remove(idx) # remove used sentences
            if y + 1 == self.train_size: # stop when enough samples
                break
        if self.cuda:
            sub = sub.cuda()

        if not values: # remove key is empty of sentences
            self.copysent.pop(self.history_len, None)
        else:
            self.copysent[self.history_len] = values # update key w left sentences
            
        return sub.view(self.history_len + 1, -1)

    def posTensor(self):
        """
        Extract the POS samples from the idx flat vector
        """
        sub = Variable(
            torch.zeros(
                self.history_len + 1,
                self.train_size, dtype=torch.long),
            requires_grad=False)

        for y, idx in enumerate(self.val4pos):
            sub[:,y] = self.pos_idx[idx:idx+self.history_len+1]
            if y + 1 == self.train_size: # stop when enough samples
                break
        del self.val4pos
        if self.cuda:
            sub = sub.cuda()

        return sub.view(self.history_len + 1, -1)

    def subTensor_test(self):
        """
        Extract the samples from the idx flat vector
        """
        sub = Variable(
            torch.zeros(
                self.history_len + 1,
                self.train_size, dtype=torch.long),
            requires_grad=False)
        idx = self.real_idx[0]
        sub[:,0] = self.vec[idx:idx+self.history_len+1]
        if self.cuda:
            sub = sub.cuda()
        return sub.view(self.history_len + 1, -1)

    def index2embed(self, sequence):
        """
        Replace index with embedding
        """
        row, col = sequence.size()
        new_seq1 = sequence.view(row * col, -1)
        new = self.veclist(new_seq1) # employ embedding layer
        new = new.view(row, col, self.dim)
        new = new.detach()
        if self.cuda:
            new = new.cuda()
        return new

    def labels(self):
        """
        Generate labels for D, G
        """
        if self.cuda:
            true_labels = torch.cuda.FloatTensor(
                self.train_size, 1).uniform_(0.9, 1)
            true_labels = Variable(true_labels)
            return true_labels, 1 - true_labels
        else:
            true_labels = torch.FloatTensor(
                self.train_size, 1).uniform_(
                0.9, 1)
            true_labels = Variable(true_labels)
            return true_labels, 1 - true_labels

    def sampler4D(self):
        """
        Train D on true and fake samples
        """
        for i in range(20000000):
            self.choose_len()
            self.number_of_sent()
            self.final_size()
            self.choose_window()
            data = self.subTensor()
            batch = self.index2embed(data)
            # generate labels
            true, fake = self.labels()
            for j in range(self.history_len // self.const_window + 1):  # extract all mutually exclusive possbile histories
                win = min(self.window, self.history_len - j * self.window)
                if win == 0: # window size is a factor of history
                    continue
                for w in range(1,win+1):
                    yield batch[j * self.window :j * self.window + w, :, :], batch[j * self.window + w, :, :], true, self.train_size, w
                    yield batch[j * self.window :j * self.window + w, :, :],                                   fake, self.train_size, w

    def sampler4G(self):
        """
        Train G on g (fooling D)
        """
        for i in range(20000000):
            self.choose_len()
            self.number_of_sent()
            self.final_size()
            self.choose_window()
            data = self.subTensor()
            batch = self.index2embed(data)
            true, fake = self.labels()
            for j in range(self.history_len // self.const_window + 1):  # extract all mutually exclusive possbile histories
                win = min(self.window, self.history_len - j * self.window)
                if win == 0: # window size is a factor of history
                    continue
                for w in range(1,win+1):
                    target_idx = data[j * self.window + w,:].cpu().data.numpy().astype(int)
                    yield batch[j * self.window :j * self.window + w, :, :], batch[j * self.window + w, :, :], self.train_size, target_idx, w, true, self.new
                    if self.new:
                        self.new = False


    def testSubSamplerG(self):
        """
        Subsample (from a bigger dataset) for continuous models
        """
        for f in range(20000000):
            # choose samples
            print(len(self.idxs), self.nsamples)
            nidxs = random.sample(range(len(self.idxs)-2), self.nsamples-4)
            for j in nidxs:
                self.real_idx = [self.idxs[j]] # show all from second to one before last
                self.history_len = self.idxs[j+1]- 2 - self.idxs[j] # ignore eof
                if self.history_len < 0:
                    print(self.idxs[j], self.idxs[j+1])
                    continue
                data = self.subTensor_test()
                batch = self.index2embed(data)
                if self.pos:
                    for i in range(1,self.history_len+1): 
                        yield batch[:i,:,:], self.train_size, data[i,:].cpu().data.numpy().astype(int), i, self.pos_idx[self.idxs[j]+i], True
                else:
                    for i in range(1,self.history_len+1): 
                        yield batch[:i,:,:], self.train_size, data[i,:].cpu().data.numpy().astype(int), i, None, True
            yield None, None, None, None, None, False

    def testSamplerG(self):
        """
        Provide all samples on the particular set
        """
        self.copysent = copy.deepcopy(self.keep)

        if self.pos:
            yield self.pos2int, self.int2pos

        while(self.copysent):
            self.choose_len()
            self.number_of_sent()
            self.final_size()
            data = self.subTensor()
            batch = self.index2embed(data)
            if self.pos:
                pos = self.posTensor()
                for i in range(1,self.history_len+1): 
                    yield batch[:i,:,:], self.train_size, data[i,:], i, pos[i,:]
            else:
                for i in range(1,self.history_len+1): 
                    yield batch[:i,:,:], self.train_size, data[i,:], i, None

    def sampler4Baseline(self):
        """
        Train Baseline 
        """
        for i in range(20000000):
            self.choose_len()
            self.number_of_sent()
            self.final_size()
            data = self.subTensor()
            batch = self.index2embed(data)
            # train on all at once
            for i in range(1,self.history_len+1): 
                yield batch[:i,:,:], self.train_size, data[i,:], i, self.new
                if self.new:
                    self.new = False
