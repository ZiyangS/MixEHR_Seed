import logging
import argparse
import torch
from GDTM import GDTM
from corpus import Corpus

logger = logging.getLogger("GDTM training processing")
parser = argparse.ArgumentParser()
# default arguments
parser.add_argument('corpus', help='Path to read corpus file', default='./store/')
parser.add_argument('output', help='Directory to store model', default='./result/')
parser.add_argument("-epoch", "--max_epoch", help="Maximum number of max_epochs", type=int, default=10)
parser.add_argument("-batch_size", "--batch_size", help="Batch size of a minibatch", type=int, default=1000)
parser.add_argument("-every", "--save_every", help="Store model every X number of iterations", type=int, default=1)
# we use GPU, printed result is "cuda"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def run(args):
    # print(args)
    # cmd = args.cmd
    seeds_topic_matrix = torch.load("./phecode_mapping/all_seed_topic_matrix.pt", map_location=device) # get seed word-topic mapping, V x K matrix
    print(seeds_topic_matrix.sum()) # 7718 as 7718 words are seed words across topics
    print(seeds_topic_matrix.shape)
    corpus = Corpus.read_corpus_from_directory(args.corpus)
    gdtm = GDTM(corpus, seeds_topic_matrix, args.batch_size, args.output)
    gdtm = gdtm.to(device)
    logger.info('''
    #     ======= Parameters =======
    #     mode: \t\ttraining
    #     file:\t\t%s
    #     output:\t\t%s
    #     max iterations:\t%s
    #     batch size:\t%s
    #     save every:\t\t%s
    #     ==========================
    # ''' % (args.corpus, args.output, args.max_epoch, args.batch_size, args.save_every))
    gdtm.inference_SCVB_SGD(max_epoch=args.max_epoch, save_every=args.save_every)

if __name__ == '__main__':
    run(parser.parse_args(['./test_store/', './result/']))
