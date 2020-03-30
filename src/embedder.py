import numpy as np
import torch
from torch.autograd import Variable
from collections import defaultdict
import pdb


def pos(path, embdict):
    """
    create a POS dict
    """
    pos_dict = defaultdict(set)
    for train_sent, pos_sent in zip(open(path+'train', 'r').readlines(), open(path+'train_pos', 'r').readlines()):
        train_sent = train_sent.strip()
        train_sent = train_sent.split()
        pos_sent = pos_sent.strip()
        pos_sent = pos_sent.split()
        for term, pos in zip(train_sent, pos_sent):
            try: # map a term w possible pos
                embdict[term]
                old = pos_dict[term]
                old.add(pos)
                pos_dict[term]=old
            except: # if a term is not in embedding assign pos to unk
                old = pos_dict['<unk>']
                old.add(pos)
                pos_dict[term]=old
    return pos_dict

def annoylists(corpus, embdict, vecdim):
    """
    provides lists for annoy
    """
    termlist = []
    dictln = len(corpus.dictionary)
    vec_mat = np.zeros((dictln, vecdim), dtype=float)
    for i in range(dictln):
        term = corpus.dictionary[i]
        vec = embdict[term]
        termlist.append(term)
        vec_mat[i,:] = vec.data.cpu().numpy()

    return vec_mat, termlist

def sym2vec(embd):
    """
    load embeddings 
    """
    edict = {}
    for h, line in enumerate(open(embd, 'r').readlines()):
        line = line.strip()
        line = line.split()
        word = line[0]
        vector = [float(item) for item in line[1:]]
        edict[word] = torch.FloatTensor(vector)
    return edict
