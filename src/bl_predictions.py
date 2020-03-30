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
import cupy as cp
import numpy as np
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

total = 0
cnt = 0
top1 = 0
top10 = 0
mrr = 0
type1 = cp.array([], dtype=int) 
type10 = cp.array([], dtype=int) 

for _fake_input, samp_size, target_idx, seqlen, pos in model_test:
     
    total+=samp_size

    hidden = net.init_hidden(samp_size)
    fake = net(_fake_input, hidden, seqlen)

    preds = fake.cpu().data.numpy()
    preds = cp.array(preds)
    target = cp.array(target_idx.cpu().data.numpy())
 
    pred_idxs = preds.argmax(axis=1)
    acc = cp.sum(target==pred_idxs)
    type1 = cp.concatenate((type1, target[cp.where(target==pred_idxs)]))
    try:
        type1 = cp.unique(type1)
    except:
        print(type1)
    top1+=acc

    ranks = (cp.negative(preds)).argsort(axis=1)
    ranks_of_best = cp.where(ranks==target.reshape(-1,1))[1]
    recip_ranks = 1.0 / cp.add(ranks_of_best,1)
    mrr+=float(cp.sum(recip_ranks[cp.where(recip_ranks > 0.099)[0]])) # below rank 10 it's 0
    acc10 = cp.where(ranks_of_best<10)[0]
    top10+=len(acc10)
    
    # from top10 predictions reduce target, if zero is detected then this target is in top10
    #pdb.set_trace()
    intop10 = (cp.negative(preds)).argsort(axis=1)[:,:10] - cp.repeat(target, 10).reshape(len(target), -1)
    rows = cp.where(intop10 == 0)[0]
    new = target[rows]
    type10 = cp.concatenate((type10, new))
    type10 = cp.unique(type10)

    if total > 500000:
        type1_np = cp.asnumpy(type1).tolist() 
        type10_np = cp.asnumpy(type10).tolist() 
        # make dict
        d = {'total': total, 'top1': int(top1), 'top10': int(top10), 'mrr': mrr, 'type1': type1_np, 'type10': type10_np}
        # save
        with open(args.name+'/preds/results_'+str(cnt)+'.txt', 'w') as outfile:
            json.dump(d, outfile)
        top1 = 0
        top10 = 0
        mrr = 0
        type1 = cp.array([], dtype=int) 
        type10 = cp.array([], dtype=int) 
        cnt+=1
        total = 0

d = {'total': total, 'top1': int(top1), 'top10': int(top10), 'mrr': mrr, 'type1': type1_np, 'type10': type10_np}
with open(args.name+'/preds/results_'+str(cnt)+'.txt', 'w') as outfile:
    json.dump(d, outfile)
