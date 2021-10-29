import argparse
import logging
import numpy as np
import os
import sys
from corpus_GPU import Corpus
import torch
from GDTM_GPU import GDTM
from utils import tokenize_phecode_icd_corpus
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
    seeds_topic_matrix = torch.load("./phecode_mapping/seed_topic_matrix.pt", map_location=device) # get seed word-topic mapping, V x K matrix
    # print(seeds_topic_matrix.sum()) # 7718 as 7718 words are seed words across topics
    gdtm = GDTM(corpus, seeds_topic_matrix, args.output)
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
