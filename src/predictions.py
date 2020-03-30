from embedder import sym2vec, annoylists, pos
from sampler import Samples_Generator
from evaluate import evaluate_preds
import logging
import argparse
import json
import torch
import numpy as np
import pickle
import pdb
import os
import sys
import new_data
sys.path.append(os.getcwd())

#################################################################
# Parsing Arguments
#################################################################

parser = argparse.ArgumentParser(description='PyTorch GAN Language Model Game')
parser.add_argument('--save', type=str, help='path to save the final Generator model')
parser.add_argument('--data', type=str, default="../orig_sets/set_0/", help='location of the data corpus')
parser.add_argument('--name', type=str, default='gan', help='the name of the model')
parser.add_argument('--pretrained', type=str, default="../embeddings/glove_50_norm", help='location of the pretrained embeddings')
parser.add_argument('--emdim', type=int, default=50, help='Embedding size')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--pos', default=False, action='store_true', help='decode based on POS')
parser.add_argument('--sample_size', type=int, default=100, help='sampel size of set')
args = parser.parse_args()


use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = torch.cuda.current_device()

if use_cuda:
    print("CUDA is up")
    torch.cuda.manual_seed(args.seed)
else:
   print("no cuda")
   sys.exit()

#################################################################
# Load model
#################################################################

if use_cuda:
    with open(args.save, 'rb') as f:
        net = torch.load(f, map_location='cuda')
else:
    with open(args.save, 'rb') as f:
        net = torch.load(f, map_location='cpu')
    
#################################################################
# Load data
#################################################################

embdict = sym2vec(args.pretrained)

with open(args.data + 'corpusG.pickle', 'rb') as f:
    corpus = pickle.load(f)

veclist, _ = annoylists(corpus, embdict, args.emdim)

#################################################################
# Dynamic Batching code
#################################################################

model_test = Samples_Generator(corpus.test, args.sample_size, args.emdim, veclist, use_cuda, args.pos, args.data+'test', mode='test').testSamplerG()

del embdict
del corpus

for p in net.parameters():
    p.requires_grad = False  # no grads

if args.pos:
    pos2int, int2pos = next(model_test) # dicts for pos decoding
    with open(args.name+'/int2pos', 'w') as fp:
        json.dump(int2pos, fp)

flag = True
total = 0
cnt = 0
for _fake_input, samp_size, target_idx, seqlen, pos in model_test:

    hidden = net.init_hidden(samp_size)
    fake = net(_fake_input, hidden, seqlen)

    if flag:
        flag = False
        preds = fake
        tgt = target_idx
        poss = pos
    else:
        preds = torch.cat((preds, fake), 0)
        tgt = torch.cat((tgt, target_idx),0)
        if args.pos:
            poss = torch.cat((poss, pos),0)
    total+=samp_size
    if total > 500000:
        # save
        torch.save(preds, args.name+'/preds'+'/preds_'+str(cnt)+'.pt')
        torch.save(tgt, args.name+'/preds'+'/tgt_'+str(cnt)+'.pt')
        if args.pos:
            torch.save(poss, args.name+'/preds'+'/pos_'+str(cnt)+'.pt')
        cnt+=1
        total = 0
        flag = True

torch.save(preds, args.name+'/preds'+'/preds_'+str(cnt)+'.pt')
torch.save(tgt, args.name+'/preds'+'/tgt_'+str(cnt)+'.pt')
if args.pos:
    torch.save(poss, args.name+'/preds'+'/pos_'+str(cnt)+'.pt')
