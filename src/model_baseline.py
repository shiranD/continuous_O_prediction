import torch.nn as nn
import torch
from torch.autograd import Variable


class Baseline(nn.Module):

    """Container module with an encoder, a recurrent module, and a decoder."""
    def __init__(self, rnn_type, emb_dim, nhid, nlayers, ntoken, loss_type):
        super(Baseline, self).__init__()
        self.rnn_type = rnn_type
        if self.rnn_type == 'LSTM': 
            self.rnn = nn.LSTM(emb_dim, nhid, nlayers)
        if self.rnn_type == 'GRU': 
            self.rnn = nn.GRU(emb_dim, nhid, nlayers)
        self.loss_type = loss_type
        if self.loss_type == 'nce':
            uni_dist = torch.distributions.uniform.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
            unigram_prob = uni_dist.sample(torch.Size([ntoken])).squeeze()
            self.decoder = linear_nce(nhid, ntoken, unigram_prob)
        else:
            self.decoder = nn.Linear(nhid , ntoken) # nhid*nlayers*batch_size / emb_dim
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers
        self.emb_dim = emb_dim

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, len_seq, targets=None, mode='train'):
        output, hiddend = self.rnn(input, hidden)
        output = output[len_seq-1, :, :]
        decoded = self.decoder(output)
        return decoded 

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM': 
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        if self.rnn_type == 'GRU': 
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


class Baseline_Generator(nn.Module):

    """Container module with an encoder, a recurrent module, and a decoder."""
    def __init__(self, rnn_type, emb_dim, nhid, nlayers):
        super(Baseline_Generator, self).__init__()
        self.rnn_type = rnn_type
        if self.rnn_type == 'LSTM': 
            self.rnn = nn.LSTM(emb_dim, nhid, nlayers)
        if self.rnn_type == 'GRU': 
            self.rnn = nn.GRU(emb_dim, nhid, nlayers)
        self.decoder1 = nn.Linear(nhid , emb_dim) # nhid*nlayers*batch_size / emb_dim
        self.decoder2 = nn.Tanh() # predict an embedding
        self.init_weights()
        self.noise = 3
        self.nhid = nhid
        self.nlayers = nlayers
        self.emb_dim = emb_dim

    def init_weights(self):
        initrange = 0.1
        self.decoder1.bias.data.fill_(0)
        self.decoder1.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, len_seq):
        output, hiddend = self.rnn(input, hidden)
        output = output[len_seq-1, :, :]
        decoded1 = self.decoder1(output)
        decoded2 = self.decoder2(decoded1)
        norm_d2 = decoded2.norm(p=2, dim=1, keepdim=True)
        decoded_norm = decoded2.div(norm_d2.expand_as(decoded2))
        return decoded_norm

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM': 
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        if self.rnn_type == 'GRU': 
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

