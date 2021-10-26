import argparse
import logging
import numpy as np
import os
import sys
from corpus_GPU import Corpus
import torch
from GDTM_GPU import GDTM
from utils import tokenize_phecode_icd
import pickle

logger = logging.getLogger("GDTM training processing")
parser = argparse.ArgumentParser()
# default arguments
parser.add_argument('num_topics', help='Number of topics') # it will not be useful as we use phecode to guide topic modelling
parser.add_argument('corpus', help='Path to read corpus file', default='./store/')
parser.add_argument('output', help='Directory to store model', default='./result/')
parser.add_argument("-epoch", "--max_epoch", help="Maximum number of iterations (Default 500)", type=int, default=10)
parser.add_argument("-every", "--save_every", help="Store model every X number of iterations (Default 50)", type=int, default=50)
# arguments for pytorch
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--clip', type=float, default=0.0, help='gradient clipping')
parser.add_argument('--wdecay', type=float, default=1.2e-6, help='some l2 regularization')

# should use GPU here, printed result is "cuda"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def run(args):
    print(args)
    # cmd = args.cmd
    corpus = Corpus.read_corpus_from_directory(args.corpus)
    phecode_ids = pickle.load(open("./mapping/phecode_ids.pkl", "rb"))
    vocab_ids = pickle.load(open("./mapping/vocab_ids.pkl", "rb"))
    tokenized_phecode_icd = pickle.load(open("./mapping/tokenized_phecode_icd.pkl", "rb"))
    # phecode_ids: key is phecode, value is the mapped index of phecode from 1 to K-1
    # vocab_ids: key is icd, value is the mapped index of icd from 1 to V-1
    # tokenized_phecode_icd: key is mapped phecode index, value is mapped icd code index
    K = len(tokenized_phecode_icd.keys())
    icd_list = vocab_ids.keys()
    V = len(icd_list)
    print(K, V)
    mapping_icd_list = []
    for w_l in tokenized_phecode_icd.values():
        mapping_icd_list.extend(w_l)
    print(len(mapping_icd_list))
    seeds_topic_matrix = torch.zeros(V, K, dtype=torch.int)
    for k, w_l in tokenized_phecode_icd.items():
        for w in w_l:
            seeds_topic_matrix[w, k] = 1 # todo: every time is different, random tokenization? we should only do one tokenizationa at corpus.py
    print(seeds_topic_matrix.sum())
    gdtm = GDTM(K, corpus, seeds_topic_matrix, args.output)
    gdtm = gdtm.to(device)
    # logger.info('''
    #     ======= Parameters =======
    #     mode: \t\ttraining
    #     file:\t\t%s
    #     output:\t\t%s
    #     num topics:\t\t%s
    #     max iterations:\t%s
    #     save every:\t\t%s
    #     ==========================
    # ''' % (args.corpus, args.output, args.num_topics, args.max_iter, args.save_every))
    gdtm.inference_SCVB_SGD(args, max_epoch=args.max_epoch, save_every=args.save_every)

if __name__ == '__main__':
    run(parser.parse_args(['100', './store/', './result/']))
