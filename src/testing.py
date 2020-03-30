from embedder import sym2vec, annoylists, pos
from sampler import Samples_Generator
from evaluate import evaluate_preds
import argparse
import torch
import numpy as np
import pickle
import pdb
import os
import sys
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
parser.add_argument('--core', type=str, help='main architecture (traditional, gan, continuous)')
parser.add_argument('--sample_size', type=int, default=100, help='sampel size of set')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = torch.cuda.current_device()

if use_cuda:
    print("CUDA is up")
    torch.cuda.manual_seed(args.seed)

#################################################################
# Load model
#################################################################

if use_cuda:
    with open(args.save, 'rb') as f:
        net = torch.load(f, map_location='cuda')
        print("loaded to gpu")
else:
    with open(args.save, 'rb') as f:
        net = torch.load(f, map_location='cpu')
    
#################################################################
# Load data
#################################################################

with open(args.data + 'corpusG.pickle', 'rb') as f:
    corpus = pickle.load(f)

embdict = sym2vec(args.pretrained)

veclist, termlist = annoylists(corpus, embdict, args.emdim)

if args.pos:
    pos_dict = pos(args.data, embdict)
    from evaluate_pos import evaluate_preds

if args.core == 'traditional':
    args.pos=False # ignore pos arg
    from evaluate_traditional import evaluate_preds

#################################################################
# Dynamic Batching code
#################################################################

model_test = Samples_Generator(corpus.test, args.sample_size, args.emdim, veclist, use_cuda, args.pos, args.data+'test', mode='test').testSamplerG()

for p in net.parameters():
    p.requires_grad = False  # no grads

# evaluate the predictions
if args.core == 'traditional':
    new_result,_ = evaluate_preds(model_test, net,          termlist, None, use_cuda, 'out/monitor_'+args.name+'_test', False)
elif args.pos:
    new_result,_ = evaluate_preds(model_test, net, veclist, termlist, None, args.emdim, 'out/monitor_'+args.name+'_test', False, pos_dict)
else:
    new_result,_ = evaluate_preds(model_test, net, veclist, termlist, None, args.emdim, 'out/monitor_'+args.name+'_test', False)
