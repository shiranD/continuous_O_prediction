import os, sys
sys.path.append(os.getcwd())
import shutil
import random
import pdb
import pickle
import numpy as np
import torch
import torch.optim as optim
from embedder import sym2vec, annoylists 
from sampler import Samples_Generator
from model_baseline import Baseline
import new_data
from evaluate_traditional import evaluate_preds
import argparse
import logging
from tensorboardX import SummaryWriter

###############################################################################
# Parsing Arguments
###############################################################################

parser = argparse.ArgumentParser(description='PyTorch WGAN Language Model Game')
parser.add_argument('--save', type=str, default='baseline.pt', help='path to save the final Discriminator model')
parser.add_argument('--data', type=str, default="../orig_sets/set_0/", help='location of the data corpus')
parser.add_argument('--pretrained', type=str, default="../embeddings/glove_50_norm", help='location of the pretrained embeddings')
parser.add_argument('--iters', type=int, default=200000000, help='number of overall iterations')
parser.add_argument('--critic_iters', type=int, default=1, help='number of critic iterations')
parser.add_argument('--emdim', type=int, default=50, help='Embedding size')
parser.add_argument('--nhid', type=int, default=200, help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
parser.add_argument('--rnn_type', type=str, default='GRU', help='type of recurrent net (currently supports GRU)')
parser.add_argument('--seqlen', type=int, default=90, help='sequence length')
parser.add_argument('--log', type=str, default='log.txt', help='logger')
parser.add_argument('--loss', type=str, default='CrossEntropy', help='loss type')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--name', type=str, default='traditional', help='name of experiment')
parser.add_argument('--sample_size', type=int, default=100, help='sampel size of set')
parser.add_argument('--update', type=int, default=100000, help='iterate over this number before evaluate')
parser.add_argument('--tbrd', type=str, default='runs', help='tensor board dirname')
parser.add_argument('--jobnum', type=str, help='SLURM job id')
args = parser.parse_args()

logging.basicConfig(filename=args.log,level=logging.INFO)

logging.info("DATA: "+args.data)
logging.info("EMBEDDING: "+args.pretrained)
logging.info("MODEL NAME: "+args.save)
logging.info("EMBD DIM: "+str(args.emdim))
logging.info("LOG: "+args.log)
logging.info("LOSS: "+args.loss)
logging.info("# LAYERS: "+str(args.nlayers))
logging.info("RNN TYPE: "+args.rnn_type)
logging.info("MONITOR FILES: "+'reports/monitor_'+args.name+'_valid')
logging.info("JOB NUMBER: "+args.jobnum)
logging.info("TENSORBOARD PATH: "+args.tbrd+'/'+args.log)

writer = SummaryWriter(args.tbrd+'/'+args.log)

use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu=torch.cuda.current_device()

if use_cuda:
    logging.info("CUDA is up")
    torch.cuda.manual_seed(args.seed)
else:
    print("Not CUDA")
    #sys.exit()
embdict = sym2vec(args.pretrained)

try:
    with open(args.data+'corpusGp.pickle', 'rb') as f:
        corpus = pickle.load(f)
except:
    corpus = new_data.Corpus(args.data, embdict)
    with open(args.data+'corpusGp.pickle', 'wb') as f:
        pickle.dump(corpus,f)

veclist, termlist = annoylists(corpus, embdict, args.emdim)

###############################################################################
# Build the model
###############################################################################

baseline = Baseline(args.rnn_type, args.emdim, args.nhid, args.nlayers, len(corpus.dictionary), args.loss)
loss = torch.nn.CrossEntropyLoss()

if use_cuda:
    baseline = baseline.cuda(gpu)
    loss = loss.cuda()

optimizer = optim.Adam(baseline.parameters(), lr=1e-4, betas=(0.9, 0.999))

###############################################################################
# Dynamic Batching code
###############################################################################

baseline_train = Samples_Generator(corpus.valid, args.sample_size, args.emdim, veclist, use_cuda, False, None, 'train').sampler4Baseline()
baseline_valid = Samples_Generator(corpus.valid, args.sample_size, args.emdim, veclist, use_cuda).testSubSamplerG()

del embdict
del corpus
result = 100
i=0

for iteration in range(args.iters):

    if iteration%args.update == 2:

        for p in baseline.parameters():
            p.requires_grad = True
    baseline.zero_grad()

    _context, samp_size, true_idxs, seqlen, new_round = next(baseline_train)
    hidden = baseline.init_hidden(samp_size)
    preds = baseline(_context, hidden, seqlen) 
    err_baseline = loss(preds, true_idxs)

    # backprop
    err_baseline.backward()
    optimizer.step()

    baseline_cost = err_baseline.mean() # just for save purpose

    if new_round:
        writer.add_text("new_round", str(i), iteration)    
        i+=1
        if i == 2:
            sys.exit()
    writer.add_scalar("training_loss",float(baseline_cost.cpu().data.numpy()), iteration)

    if iteration%args.update == 1:

        for p in baseline.parameters():
            p.requires_grad = False
        baseline.zero_grad()

        # evaluate the predictions
        new_result, writer = evaluate_preds(baseline_valid, baseline, termlist, iteration, use_cuda, 'out/monitor_'+args.name+'_valid', writer)
        writer.add_scalar("valid_loss", new_result, iteration)

        # save models
        if new_result < result:
            with open(args.save, 'wb') as f:
                torch.save(baseline, f)
            result = new_result
            writer.add_scalar("saved_valid_loss", result, iteration)
            shutil.move('out/monitor_'+args.name+'_valid', 'reports/monitor_'+args.name+'_valid')
