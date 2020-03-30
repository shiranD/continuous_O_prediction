from embedder import sym2vec, annoylists, pos
import json
import pickle
import argparse
import sys
from space import build_space
import torch 
import numpy as np
import pdb
from collections import defaultdict

parser = argparse.ArgumentParser(description='evaluate predictions')
parser.add_argument('--data', type=str, default="../orig_sets/set_0/", help='location of the data corpus')
parser.add_argument('--name', type=str, default='gan', help='the name of the model')
parser.add_argument('--number', type=str, help='file number to process')
parser.add_argument('--pretrained', type=str, default="../embeddings/glove_50_norm", help='location of the pretrained embeddings')
parser.add_argument('--emdim', type=int, default=50, help='Embedding size')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--pos', default=True, action='store_true', help='decode based on POS')
args = parser.parse_args()

def evaluate_preds(folder, veclist, termlist, emdim, pos_dict, number, int2pos):

    space = build_space(emdim, veclist)
    space.build(100)
    num_vecs = len(veclist)
    max_rank = 11
    just_in_case_max_rank = 50 
    predir = '/preds'
    acc = 0
    recall = 0
    mrr = 0
    total = 0
    type1 = defaultdict(int)
    type10 = defaultdict(int)

    if args.pos:
        pos_dict[">"] = ">"
        poss = torch.load(folder+predir+'/pos_'+number+'.pt', map_location=torch.device('cpu'))
        acc_pos = 0
        recall_pos = 0
        mrr_pos = 0
        type1_pos = defaultdict(int)
        type10_pos =  defaultdict(int)

    # load files (pred, target, pos) to cpu
    preds = torch.load(folder+predir+'/preds_'+number+'.pt', map_location=torch.device('cpu'))
    targets = torch.load(folder+predir+'/tgt_'+number+'.pt', map_location=torch.device('cpu'))

    for pred, target_idx, pos in zip(preds, targets, poss):

        if args.pos:
            pos = int2pos[str(int(pos))]
        # find its nearest neighbor
        nearest, dist= space.get_nns_by_vector(pred, n=just_in_case_max_rank, search_k=-1, include_distances=True)

        # No POS decoding
        rank1 = -1
        for g, friend in enumerate(
                nearest[:10]):
            if friend == target_idx:
                rank1 = g + 1
                type10[termlist[target_idx]] +=1
                break
    
        # compute and report
        if rank1 != -1:
            recall += 1
            if rank1 == 1:
                acc += 1
                type1[termlist[target_idx]] +=1
            mrr += 1 / rank1
        
        # POS decoding
        if args.pos:
            rank = -1
            g = 0
            target_pos = pos
            for friend in nearest:
                friend_pos = pos_dict[termlist[friend]]
                if target_pos in friend_pos or not bool(friend_pos):
                    if friend == target_idx:
                        rank = g + 1
                        type10_pos[termlist[target_idx]] +=1
                    g+=1
                    if g == 10:
                        break
            if rank == -1 and rank1 > -1:
                print(rank1, rank, termlist[target_idx], pos_dict[termlist[target_idx]],pos)

            if rank != -1:
                recall_pos += 1
                if rank == 1:
                    acc_pos += 1
                    type1_pos[termlist[target_idx]] +=1
                mrr_pos += 1 / rank

        total+=1

    # to aggragate types store them
    with open(args.name+'/results/type1_'+number, 'w') as fp:
        json.dump(type1, fp)
    with open(args.name+'/results/type10_'+number, 'w') as fp:
        json.dump(type10, fp)
    fname = open(folder+'/results/summary_'+number, 'w')
    fname.write(
        "total of {} tokens with top1 of {} mrr of {:2.4f} top10 of {}\n".format( 
            total,
            acc,
            mrr, 
            recall))
    fname.close()

    if args.pos:
        with open(args.name+'/results/type1_pos_'+number, 'w') as fp:
            json.dump(type1_pos, fp)
        with open(args.name+'/results/type10_pos_'+number, 'w') as fp:
            json.dump(type10_pos, fp)
        fname_pos = open(folder+'/results/summary_pos_'+number, 'w')
        fname_pos.write(
            "total of {} tokens with top1 of {} mrr of {:2.4f} top10 of {}\n".format( 
                total,
                acc_pos,
                mrr_pos, 
                recall_pos))
        fname_pos.close()

    sys.stdout.flush()


with open(args.data + 'corpusG.pickle', 'rb') as f:
    corpus = pickle.load(f)

embdict = sym2vec(args.pretrained)

veclist, termlist = annoylists(corpus, embdict, args.emdim)

int2pos = None
if args.pos:
    pos_dict = pos(args.data, embdict)
    with open(args.name+'/int2pos', 'r') as fp:
        int2pos = json.load(fp)

evaluate_preds(args.name, veclist, termlist, args.emdim, pos_dict, args.number, int2pos)
