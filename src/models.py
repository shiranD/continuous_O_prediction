import torch.nn as nn
from torch.autograd import Variable
import torch

class Generator(nn.Module):

    """Container module with an encoder, a recurrent module, and a decoder."""
    def __init__(self, rnn_type, emb_dim, nhid, nlayers):
        super(Generator, self).__init__()
        self.rnn_type = rnn_type
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(emb_dim, nhid, nlayers)
        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(emb_dim, nhid, nlayers)
        self.decoder1 = nn.Linear(nhid , emb_dim) # nhid*nlayers*batch_size / emb_dim
        self.decoder2 = nn.Tanh() # predict an embedding
        self.init_weights()
        self.noise = 3
        self.rnn_type = rnn_type
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
            a = weight.new(self.nlayers, bsz, self.noise).normal_(-0.01,0.01)
            b = weight.new(self.nlayers, bsz, self.nhid-self.noise).zero_()
            return (Variable(torch.cat([a,b], 2)),Variable(torch.cat([a,b], 2)))
        if self.rnn_type == 'GRU': 
            a = weight.new(self.nlayers, bsz, self.noise).normal_(-0.01,0.01)
            b = weight.new(self.nlayers, bsz, self.nhid-self.noise).zero_()
            return Variable(torch.cat([a,b], 2))

class Discriminator(nn.Module):

    """Container module with an encoder, a recurrent module, and a decoder."""
    def __init__(self, rnn_type, emb_dim, nhid, nlayers, final):
        super(Discriminator, self).__init__()
        self.rnn_type = rnn_type
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(emb_dim, nhid, nlayers)
        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(emb_dim, nhid, nlayers)
        self.decoder1 = nn.Linear(nhid, emb_dim) # nhid*nlayers*batch_size / emb_dim
        self.decoder2 = nn.Tanh() # predict an embedding
        self.decoder3 = nn.Linear(emb_dim, 1)
        if final == 'Sigmoid':
            self.decoder4 = nn.Sigmoid()
        elif final == 'positive':
            self.decoder4 = nn.Softplus()
        else:
            self.decoder4 = nn.Tanh()
        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.emb_dim = emb_dim

    def init_weights(self):
        initrange = 0.1
        self.decoder1.bias.data.fill_(0)
        self.decoder1.weight.data.uniform_(-initrange, initrange)
        self.decoder3.bias.data.fill_(0)
        self.decoder3.weight.data.uniform_(-initrange, initrange)

    def forward(self, input1, hidden, input2, len_seq):
        output, hiddend = self.rnn(input1, hidden)
        output = output[len_seq-1, :, :]
        decoded1 = self.decoder1(output)
        decoded2 = self.decoder2(decoded1)
        norm_d2 = decoded2.norm(p=2, dim=1, keepdim=True)
        decoded_norm = decoded2.div(norm_d2.expand_as(decoded2))
        residual = decoded_norm - input2
        decoded4 = self.decoder3(residual) # a final score of D
        decoded5 = self.decoder4(decoded4) # a final score of D
        return decoded5 

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM': 
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        if self.rnn_type == 'GRU': 
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

