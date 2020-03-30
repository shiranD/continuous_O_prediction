import os, sys
sys.path.append(os.getcwd())
import shutil
import random
import pdb
import pickle
import numpy as np
import torch
import torch.optim as optim
from embedder import sym2vec, annoylists, pos 
from sampler import Samples_Generator
import new_data
from loss import SIMLoss_sqrt
from models import Generator, Discriminator
from evaluate import evaluate_preds
import argparse
import logging
from tensorboardX import SummaryWriter

###############################################################################
# Parsing Arguments
###############################################################################

parser = argparse.ArgumentParser(description='PyTorch WGAN Language Model Game')
parser.add_argument('--saveD', type=str, default='Disc.pt', help='path to save the final Discriminator model')
parser.add_argument('--saveG', type=str, default='Gen.pt' ,help='path to save the final Generator model')
parser.add_argument('--data', type=str, default="../orig_sets/set_0/", help='location of the data corpus')
parser.add_argument('--pretrained', type=str, default="../embeddings/glove_50_norm", help='location of the pretrained embeddings')
parser.add_argument('--iters', type=int, default=200000000, help='number of overall iterations')
parser.add_argument('--critic_iters', type=int, default=1, help='number of critic iterations')
parser.add_argument('--emdim', type=int, default=50, help='Embedding size')
parser.add_argument('--nhid', type=int, default=200, help='number of hidden units per layer')
parser.add_argument('--name', type=str, default='gan', help='the LM name')
parser.add_argument('--loss', type=str, default='CrossEntropy', help='loss type')
parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
parser.add_argument('--rnn_type', type=str, default='GRU', help='type of recurrent net (currently supports GRU)')
parser.add_argument('--seqlen', type=int, default=90, help='sequence length')
parser.add_argument('--log', type=str, default='log.txt', help='logger')
parser.add_argument('--pos', action='store_true', default=False, help='decode based on POS')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--sample_size', type=int, default=100, help='sampel size of set')
parser.add_argument('--update', type=int, default=100000, help='iterate over this number before evaluate')
parser.add_argument('--tbrd', type=str, default=False, help='tensorboard dirname')
parser.add_argument('--jobnum', type=str, help='SLURM job id')
parser.add_argument('--more', action='store_true', default=False, help='continue train or create a new model')
parser.add_argument('--origG', type=str, help='location of the trained model to adapt')
parser.add_argument('--origD', type=str, help='location of the trained model to adapt')
args = parser.parse_args()

logging.basicConfig(filename=args.log,level=logging.INFO)

logging.info("DATA: "+args.data)
logging.info("EMBEDDING: "+args.pretrained)
logging.info("MODEL NAME: "+args.saveG)
logging.info("EMBD DIM: "+str(args.emdim))
logging.info("LOG: "+args.log)
logging.info("POS: "+str(args.pos))
logging.info("LOSS: "+args.loss)
logging.info("# LAYERS: "+str(args.nlayers))
logging.info("RNN TYPE: "+args.rnn_type)
logging.info("MONITOR FILES: "+'reports/monitor_'+args.name+'_valid/_pos')
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
    sys.exit()

embdict = sym2vec(args.pretrained)

try:
    with open(args.data+'corpusG.pickle', 'rb') as f:
        corpus = pickle.load(f)
except:
    corpus = new_data.Corpus(args.data, embdict)
    with open(args.data+'corpusG.pickle', 'wb') as f:
        pickle.dump(corpus,f)

veclist, termlist = annoylists(corpus, embdict, args.emdim)

if args.pos:
    pos_dict = pos(args.data, embdict)
    from evaluate_pos import evaluate_preds

###############################################################################
# Build the model
###############################################################################

lossGAN = torch.nn.MSELoss() # construct loss

if args.loss == 'mse':
    lossSIM = torch.nn.MSELoss()
elif args.loss == 'cos':
    lossSIM = SIMLoss_sqrt(dim=1)
elif args.loss == 'MM' or args.loss=='mm':
    lossSIM = MaxMarginLoss(args.emdim, veclist, dim=1)
else:
   print('no loss function was chosen')
   sys.exit()

if use_cuda:    
    lossSIM = lossSIM.cuda()
    lossGAN = lossGAN.cuda()

if use_cuda and args.more:
    try:
        with open(args.origG, 'rb') as f:
            netG = torch.load(f, map_location='cuda')
    except ValueError:
        print("no original generator was found")
    try:
        with open(args.origD, 'rb') as f:
            netD = torch.load(f, map_location='cuda')
    except ValueError:
        print("no original descriminator was found")
else:                     # make a new model
    final='Sigmoid'
    netG = Generator(args.rnn_type, args.emdim, args.nhid, args.nlayers)
    netD = Discriminator(args.rnn_type, args.emdim, args.nhid, args.nlayers, final)
    
    if use_cuda:
        netD = netD.cuda(gpu)
        netG = netG.cuda(gpu)

# Adaptive moment distribution (momentum (Vdw_corr(b1) - exponentially weighted avg)
# and rmpsprop (divide by the square root of Sdw_corr(b2)))
optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.9, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.9, 0.999))

###############################################################################
# Dynamic Batching code
###############################################################################

D_train = Samples_Generator(corpus.train, args.sample_size, args.emdim, veclist, use_cuda, mode='train').sampler4D()
G_train = Samples_Generator(corpus.train, args.sample_size, args.emdim, veclist, use_cuda, mode='train').sampler4G()

baseline_valid = Samples_Generator(corpus.valid, args.sample_size, args.emdim, veclist, use_cuda, args.pos, args.data+'valid', mode='eval').testSubSamplerG()

del embdict
del corpus

result = 100
#reg_lambda = 0.0001
reg_lambda = 0.001
i=0

for iteration in range(args.iters):
    ############################
    # (1) Update D network
    ###########################
    for p in netD.parameters(): 
        p.requires_grad = True  # and build a computational graph

    for p in netG.parameters(): 
        p.requires_grad = False # don't build for G 

    for iter_d in range(args.critic_iters):
        netD.zero_grad() # reset previous grads

        # train with real
        _data, real, true_labels, samp_size, seqlen = next(D_train)
        hiddenD = netD.init_hidden(samp_size)
        D_real = netD(_data, hiddenD, real, seqlen)


        # reg
        l2_reg = None
        for w in netD.parameters(): 
            if l2_reg is None:
                l2_reg = w.norm(2)
            else:
                l2_reg = l2_reg + w.norm(2) 

        #r, c, dim = D_real.shape 
        err_real = lossGAN(D_real, true_labels) + l2_reg * reg_lambda
        err_real.backward()
   
        # generate fakes
        _fake_input, fake_labels, samp_size, seqlen = next(D_train)
        hiddenG = netG.init_hidden(samp_size)
        fake = netG(_fake_input, hiddenG, seqlen)
 
        # train with fake
        hiddenD = netD.init_hidden(samp_size)
        D_fake = netD(_fake_input, hiddenD, fake, seqlen)


        # reg
        l2_reg = None
        for w in netD.parameters(): 
            if l2_reg is None:
                l2_reg = w.norm(2)
            else:
                l2_reg = l2_reg + w.norm(2) 

        #r, c, dim = D_fake.shape 
        err_fake = lossGAN(D_fake, fake_labels) + l2_reg * reg_lambda
        err_fake.backward()

        # apply gradients of both
        optimizerD.step()

        D_cost = err_real.mean() + err_fake.mean()  # just for save purpose

    ############################
    # (2) Update G network 
    ###########################
    for p in netD.parameters():
        p.requires_grad = False  # to avoid making a computation graph

    for p in netG.parameters(): 
        p.requires_grad = True # build for G 
    netG.zero_grad() # avoid updating with current accumulated gradients

    #pdb.set_trace()
    _fake_input, true_embeddings, samp_size, _, seqlen, trueD_labels, new_round = next(G_train)
    hiddenG = netG.init_hidden(samp_size)
    fake = netG(_fake_input, hiddenG, seqlen) 

    # concat history + fake
    hiddenD = netD.init_hidden(samp_size)
     # get predictions
    G = netD(_fake_input, hiddenD, fake, seqlen)

    # backprop
    err_GAN = lossGAN(G, trueD_labels)
    err_EMB = lossSIM(fake, true_embeddings)

    l2_reg = None
    for w in netG.parameters(): 
        if l2_reg is None:
            l2_reg = w.norm(2)
        else:
            l2_reg = l2_reg + w.norm(2) 
    err_G = err_GAN + err_EMB + l2_reg * reg_lambda

    if new_round:
        writer.add_text("new_round", str(i) , iteration)
        i+=1

    err_G.backward() # fake labels are real for the generator
    optimizerG.step()

    G_cost = err_G.mean() # just for save purpose
    writer.add_scalar("training_loss",float(G_cost.cpu().data.numpy()), iteration)

    if iteration%args.update == 1:

        for p in netG.parameters(): 
            p.requires_grad = True # build for G 
        netG.zero_grad() # avoid updating with current accumulated gradients

        # save models
        writer.add_scalar("D_loss_real", float(err_real.mean().cpu().data.numpy()), iteration)
        writer.add_scalar("D_loss_fake", float(err_fake.mean().cpu().data.numpy()), iteration)
        writer.add_scalar("G_loss_emd",  float(err_EMB.mean().cpu().data.numpy()), iteration)
        writer.add_scalar("G_loss_D",    float(err_GAN.mean().cpu().data.numpy()), iteration)
        writer.add_scalar("D_loss" ,     float(D_cost.mean().cpu().data.numpy()), iteration)
        writer.add_scalar("G_loss",      float(G_cost.mean().cpu().data.numpy()), iteration)

        # evaluate the predictions
        if args.pos:
            new_result, writer = evaluate_preds(baseline_valid, netG, veclist, termlist, iteration, args.emdim, 'out/monitor_'+args.name+'_valid', writer, pos_dict)
        else:
            new_result, writer = evaluate_preds(baseline_valid, netG, veclist, termlist, iteration, args.emdim, 'out/monitor_'+args.name+'_valid', writer)

        # save models
        writer.add_scalar("valid_loss", new_result, iteration)
        if new_result < result:
            with open(args.saveD, 'wb') as f:
                torch.save(netD, f)
            with open(args.saveG, 'wb') as f:
                torch.save(netG, f)
            result = new_result
            writer.add_scalar("saved_valid_loss", result, iteration)
            shutil.move('out/monitor_'+args.name+'_valid', 'reports/monitor_'+args.name+'_valid')
            shutil.move('out/monitor_'+args.name+'_valid_pos', 'reports/monitor_'+args.name+'_valid_pos')
