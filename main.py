import argparse
import logging
import numpy as np
import os
import sys

logger = logging.getLogger("GDTM training processing")
parser = argparse.ArgumentParser()
# default arguments
parser.add_argument('num_topics', help='Number of topics')
parser.add_argument('corpus', help='Path to read corpus file', default='./store/')
parser.add_argument('output', help='Directory to store model', default='./result/')
parser.add_argument("-iter", "--max_iter", help="Maximum number of iterations (Default 500)", type=int, default=500)
parser.add_argument("-every", "--save_every", help="Store model every X number of iterations (Default 50)",
                            type=int, default=50)


def run(args):
    cmd = args.cmd
    # corpus = Corpus.read_corpus_from_directory(args.corpus)
    # gdtm = GDTM(int(args.num_topics), corpus, args.output)
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
    #
    # mixehr.inference_svb(max_iter=args.max_iter, save_every=args.save_every)


if __name__ == '__main__':
    run(parser.parse_args(['100', './corpus', './store/', './result/']))

