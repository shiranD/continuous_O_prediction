#!/usr/local/bin python3
# -*- coding: utf-8 -*-
import sys
from os import walk
import argparse
import json
import io
from gensim.models import Word2Vec
from collections import defaultdict
import numpy as np
import random

random.seed(9001)

def embedding(args):

    # read jsons

    sents = []
    terms = defaultdict(int)
    for line in open(args.filename, 'r').readlines():
        line = line.strip()
        nline='<S> '+line+' </S>'
        sent = []
        for token in nline.split():
            sent.append(token)
            #frequency table
            terms[token]+=1
        sents.append(sent)
    random.shuffle(sent)
    # train model
    model = Word2Vec(size=args.emdim, min_count=1)
    model.build_vocab(sents)
    for epoch in range(1):
        model.train(sents, total_examples=model.corpus_count, epochs=5)
    
    f = open(args.modelname, 'w')
    vec = np.ones(args.emdim)
    norm_vec = np.linalg.norm(vec)
    normed = vec / norm_vec # l2 norm
    vec_list = [str(i) for i in normed]
    f.write('{} {}\n'.format('<unk>', ' '.join(vec_list)))
    for word in model.wv.vocab.keys():
        vec = model[word]
        norm_vec = np.linalg.norm(vec)
        normed = vec / norm_vec # l2 norm
        vec_list = [str(i) for i in normed]
        f.write('{} {}\n'.format(word, ' '.join(vec_list)))
    f.close()

    # print freq table
    f = open(args.modelname+'_count', 'w')
    sorted_terms = sorted(terms.items(), key=lambda kv: kv[1])
    for key, val in sorted_terms:
        f.write("{} {}\n".format(key, val))
    f.write("there are {} number of sentences".format(len(sents)))
    f.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='create word2vec embeddings')
    parser.add_argument('--filename', type=str, help='dir path')
    parser.add_argument('--modelname', type=str, help='path to the model')
    parser.add_argument('--emdim', type=int, help='vector dimensions')
    args = parser.parse_args()
    embedding(args)
