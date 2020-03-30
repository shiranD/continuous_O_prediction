import sys
import os
import argparse
import pickle
import json
import torch
from collections import defaultdict

parser = argparse.ArgumentParser(description='split to old new and shared terms')
parser.add_argument('--num', type=int, help='num of files')
parser.add_argument('--data', type=str, default="../orig_sets/set_0/", help='location of the data corpus')
parser.add_argument('--outdir', type=str, help='output dir')
parser.add_argument('--name', type=str, help='name of the set')
parser.add_argument('--inference_folder', type=str, help='name of the set')
args = parser.parse_args()

with open(args.data+'corpusG.pickle', 'rb') as f:
    corpus = pickle.load(f)

d_train = corpus.train["ids"]
e = corpus.dictionary

high = defaultdict(int)
mid1 = defaultdict(int)
mid2 = defaultdict(int)
low = defaultdict(int)
oov = defaultdict(int)

high_w = defaultdict(int)
mid1_w = defaultdict(int)
mid2_w = defaultdict(int)
low_w = defaultdict(int)
oov_w = defaultdict(int)

lowf = 0
mid1f = 0
mid2f = 0
highf = 0
oovf = 0

freq = defaultdict(int)
for term in d_train:
    freq[term]+=1

predir='preds'

for number in range(args.num):
    try:
        targets = torch.load(args.inference_folder+'/'+predir+'/tgt_'+str(number)+'.pt', map_location=torch.device('cpu'))
        for term in targets: 
            term = int(term)
            if term in freq and term != 0:
                v = freq[term]
                if v < 10:
                    low[term] = True
                    low_w[e[term]] = True
                    lowf+=1
                elif v < 100:
                    mid1[term] = True
                    mid1_w[e[term]] = True
                    mid1f+=1
                elif v < 1001:
                    mid2[term] = True
                    mid2_w[e[term]] = True
                    mid2f+=1
                elif v > 1000:
                    high[term] = True
                    high_w[e[term]] = True
                    highf+=1
                else:
                    print('what ', e[term], v)
            elif term !=0: # not seen in training
                try:
                    oov[term] = True
                    oov_w[e[term]]=True
                except:
                    oov[e.word2idx['<unk>']]=True
                    oov_w['<unk>']=True
                oovf+=1
    except:
       pass
   
with open(args.outdir+"/high_"+args.name, 'w') as fp:
    json.dump([high_w, highf], fp)
with open(args.outdir+"/mid1_"+args.name, 'w') as fp:
    json.dump([mid1_w, mid1f], fp)
with open(args.outdir+"/mid2_"+args.name, 'w') as fp:
    json.dump([mid2_w, mid2f], fp)
with open(args.outdir+"/low_"+args.name, 'w') as fp:
    json.dump([low_w, lowf], fp)
with open(args.outdir+"/oov_"+args.name, 'w') as fp:
    json.dump([oov_w, oovf], fp)

with open(args.outdir+"/bl_high_"+args.name, 'w') as fp:
    json.dump([high, highf], fp)
with open(args.outdir+"/bl_mid1_"+args.name, 'w') as fp:
    json.dump([mid1, mid1f], fp)
with open(args.outdir+"/bl_mid2_"+args.name, 'w') as fp:
    json.dump([mid2, mid2f], fp)
with open(args.outdir+"/bl_low_"+args.name, 'w') as fp:
    json.dump([low, lowf], fp)
with open(args.outdir+"/bl_oov_"+args.name, 'w') as fp:
    json.dump([oov, oovf], fp)

print('high types {}'.format(len(high)))
print('m2 types {}'.format(len(mid2)))
print('m1 types {}'.format(len(mid1)))
print('low types {}'.format(len(low)))
