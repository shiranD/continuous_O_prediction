import os
import torch
from collections import defaultdict
import pdb

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def __getitem__(self, idx):
        return self.idx2word[idx]

    def to_idx(self, word):
        return self.word2idx[word]


class Corpus(object):
    def __init__(self, path, embdict):
        self.dictionary = Dictionary()
        # assert input path validity
        self.emdict = embdict
        assert os.path.exists(os.path.dirname(
            path + "train")), "%r is not a valid path" % path + "train"
        assert os.path.exists(os.path.dirname(
            path + "valid")), "%r is not a valid path" % path + "valid"
        assert os.path.exists(os.path.dirname(
            path + "test")), "%r is not a valid path" % path + "test"
        self.train = self.tokenize(path + 'train')
        self.valid = self.tokenize(path + 'valid')
        self.test = self.tokenize(path + 'test')

    def tokenize(self, path):
        """
        Tokenizes a text file
        and represent it as indecies
        """
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            # populate a dict with sent lengths
            tokens = 0
            sents = []
            for line in f:
                words = ['<S>']
                words+=line.split()
                words+=['</S>'] # this token should not be converted 
                sent = []
                for word in words:
                    try:
                        self.emdict[word]
                        self.dictionary.add_word(word)
                        sent.append(word)
                    except:
                        sent.append('<unk>')
                        self.dictionary.add_word('<unk>')
                    tokens+=1
                sents.append(sent)

        # help bring back to sentences
        # a list of tensors
        # Tokenize file content
        ids = [None]*tokens 
        len_dict = defaultdict(list)
        sent2idx = []
        token = 0
        for j, words in enumerate(sents):
            sent2idx.append(token) # position
            len_dict[len(words)-1]+=[j] # histogram of lengths
            for word in words:
                ids[token] = self.dictionary.word2idx[word]
                token += 1

        idx_dict = {}
        total = 0
        for key, vals in len_dict.items():
            idxs = []
            for val in vals:
                idxs.append(sent2idx[val])
            total+=len(idxs)
            idx_dict[key] = idxs

        d = {}
        d["ids"]=ids
        d["total"]=total
        d["len2sentidx"]=idx_dict
        return d
