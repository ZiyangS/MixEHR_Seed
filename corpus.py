# define data structure for corpus
from typing import Mapping, List, NoReturn, Set, TypeVar
import numpy as np
import pandas as pd
import pickle
import os
import logging
import sys
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='Select one command', dest='cmd')

# parser process
parser_process = subparsers.add_parser('process', help="Transform MixEHR raw data")
parser_process.add_argument("-n", "--max", help="Maximum number of observations to select", type=int, default=None)

# parser split
parser_split = subparsers.add_parser('split', help="Split data into train/test")
parser_split.add_argument("-tr", "--testing_rate", help="Testing rate. Default: 0.2", type=float, default=0.2)

# default arguments
parser.add_argument('input', help='Directory containing input data')
parser.add_argument('output', help='Directory where processed data will be stored')


class Corpus(Dataset):
    def __init__(self, patients, T, V) -> NoReturn:
        logger.info("Creating corpus...")
        self.dataset = patients
        self.D = len(patients)
        self.T = T
        self.V = V

    def __len__(self):
        return self.D

    # def BOW_representation(self):
    #     # full batch, we assume
    #     BOW = np.zeros(self.D, self.V)
    #     for doc in self.dataset:
    #         words_dict = doc.words_dict
    #         for word_id, freq in words_dict.items():
    #             BOW[doc.doc_id, word_id] = freq
    #     return BOW
    #
    # def corpus_time_labels(self):
    #     # full batch, we assume
    #     t_labels = np.zeros(self.D)
    #     for doc in self.dataset:
    #         t_labels[doc.doc_id] = doc.t_d
    #     return t_labels

    @staticmethod
    def __collate_model__(batch):
        '''
        Returns a batch for each iteration of the DataLoader
        :param batch:
        :return:
        '''
        docs, indixes = zip(*batch)
        # return list(patients), np.array(indixes), np.sum([p[0].Cj for p in batch])
        return list(docs), np.array(indixes), [doc[0].t_d for doc in batch] # return times of docs, a list

    @staticmethod
    def build_from_mixehr_fileformat(data_path: str, time_path: str, store_path: str = None,):

        def __read_time__(doc_ids, times):
            D = len(doc_ids.keys())
            t = {}
            pbar = tqdm(doc_ids.keys())
            for i, doc in enumerate(pbar):
                time = times[times['doc_id'] == doc]['age_groups'].item()
                record = doc_ids[doc]
                t[doc] = time
                pbar.set_description("%.4f  - patient(%s)" % (100 * (i + 1) / D, record))
            return t

        def __read_docs__(data, times, doc_ids, pat_ids, vocab_ids):
            training = {}
            num_records = data.shape[0]
            with tqdm(total=num_records) as pbar:
                for i, row in enumerate(data.iterrows()):
                    row = row[1]
                    doc_id = row['doc_id']
                    pat_id = row['pat_id']
                    if doc_id not in doc_ids: #?
                        continue
                    word_id = vocab_ids[row['dx']]
                    freq = row['freq']
                    if doc_id not in training:
                        training[doc_id] = Corpus.Document(doc_id, pat_id, times[doc_id],)
                    document = training[doc_id]
                    document.append_record(word_id, freq)
                    pbar.set_description("%.4f  - document(%s), patient(%s), word(%s)"
                                         % (100 * (i + 1) / num_records, doc_id, pat_id, word_id))
                    pbar.update(1)
            return training

        def __store_data__(toStore, corpus):
            print('store data....')
            if not os.path.exists(toStore):
                os.makedirs(toStore)

            corpus_file = os.path.join(toStore, "corpus.pkl")
            logger.info("Saving: \n\t%s" % (corpus_file))
            pickle.dump(corpus, open(corpus_file, "wb"))
            logger.info("Data stored in %s" % toStore)

        # read files
        data = pd.read_csv(data_path, sep='\t') # read documents data
        times = pd.read_csv(time_path) # read time of documents
        print(data)
        print(times)
        # M = data.groupby(0).sum()[3].values  # number of words in for record j
        # D = data[0].unique().shape[0]  # number of patients
        # map data ids
        doc_ids = {}
        pat_ids = {}
        vocab_ids = {}

        ids = times['doc_id'].unique() # get index of each document, it is a age group division of patient admissions
        for i, doc_id in enumerate(ids): # from 0 to D? I guess, need to test
            doc_ids[doc_id] = i
        # do we need to set a patient's record? after preprocessing data, we need to check
        unique_patients = times['pat_id'].unique()
        for i, pat_id in enumerate(unique_patients):
            pat_ids[pat_id] = i  # the patients are from 0 to P-1
        unique_words = data['dx'].unique()
        for i, word_id in enumerate(unique_words):
            vocab_ids[word_id] = i  # the words are from 0 to V-1
        # do we need to process seed word right?

        with open('mapping/doc_ids.pkl', 'wb') as handle:
            pickle.dump(doc_ids, handle)
        with open('mapping/pat_ids.pkl', 'wb') as handle:
            pickle.dump(pat_ids, handle)
        with open('mapping/vocab_ids.pkl', 'wb') as handle:
            pickle.dump(vocab_ids, handle)
        print("finish exporting")

        # Process and read documents
        t = __read_time__(doc_ids, times) # read temporal labels of documents
        dataset = __read_docs__(data, t, doc_ids, pat_ids, vocab_ids) # read documents

        # Set data to Corpus object
        T = len(set(t.values()))
        W = len(vocab_ids)
        corpus = Corpus([*dataset.values()], T, W)

        logger.info(f'''
        ========= DataSet Information =========
        Documents: {len(corpus.dataset)}
        Patients: {len(corpus.dataset)}
        Times: {corpus.T}
        Word Tokes: {corpus.W}
        ======================================= 
        ''')

        if store_path:
            __store_data__(store_path, corpus)

        return corpus

    @staticmethod
    def split_train_test(corpus, split_rate: float, toStore: str):
        assert split_rate >= .0 and split_rate <= 1., "specify the rate for splitting training and test. e.g 0.8 = 80% for testing"

        def __store_data__(toStore, corpus):
            print('store data....')
            if not os.path.exists(toStore):
                os.makedirs(toStore)
            corpus_file = os.path.join(toStore, "corpus.pkl")
            logger.info("Saving: \n\t%s" % (corpus_file))
            pickle.dump(corpus, open(corpus_file, "wb"))
            logger.info("Data stored in %s" % toStore)

        def __split__(size, corpus):
            documents = [] # initialize to store train documents
            corpus_list = [None, None]
            splitted = False
            index = 0 # set index to zero for train set, doc_id from 0 to D-1
            dbar = tqdm(corpus)
            for doc, _, in dbar:
                dbar.set_description("Processing document %s (Patient index: %s)" % (doc.doc_id, doc.pat_id))
                doc.doc_id = index # doc_id from 0 to D-1
                index += 1
                documents.append(doc)
                if index == size and not splitted:
                    corpus_list[0] = Corpus(documents, corpus.T, corpus.W) # obtain train set
                    index = 0 # set index to zero for test set
                    documents = [] # initialize to store test documents
                    splitted = True
            corpus_list[1] = Corpus(documents, corpus.T, corpus.W) # obtain test set
            return tuple(corpus_list)

        train_size = corpus.D - int(split_rate * corpus.D)
        train, test = __split__(train_size, corpus)
        # store data
        __store_data__(os.path.join(toStore, 'train'), train)
        __store_data__(os.path.join(toStore, 'test'), test)
        logger.info("Training size: %s\nTesting size: %s\n" % (train_size, corpus.D - train_size))

    @staticmethod
    def read_corpus_from_directory(path, read_metadata=False):
        '''
        Reads existed data
        :param path: folder containing corpus files
        :return: Corpus object and metadata (patient ids, data type ids, vocab ids)
        '''
        corpus_file = os.path.join(path, "corpus.pkl")
        corpus = pickle.load(open(corpus_file, "rb"))
        # precompute all labels
        time_labels = np.zeros(len(corpus))
        for doc, _ in corpus:
            time_labels[doc.doc_id] = doc.t
        corpus.labels = time_labels
        return corpus

    @staticmethod
    def generator_mini_batch(corpus, batch_size):
        generator = DataLoader(corpus, batch_size=batch_size, shuffle=True, collate_fn=Corpus.__collate_model__)
        return generator

    @staticmethod
    def generator_full_batch(corpus):
        generator = DataLoader(corpus, batch_size=len(corpus), shuffle=True, collate_fn=Corpus.__collate_model__)
        return generator


    class Document(object):
        def __init__(self, doc_id, pat_id, t: int, words_dict: dict = None):
            '''
            Create a new patient.
            '''
            self.words_dict = words_dict if words_dict is not None else {} # key: word_id, value: frequency
            self.doc_id = doc_id # index the order of document,  doc_id for train set and test set starts from 0
            self.pat_id = pat_id # index patient, it is a part of original data
            self.t_d = int(t) # t_d, the temporal label for each document, in this case, it is the age group
            # Cj is the length of document, which is the sum of each word frequency
            # self.Cj = Cj

        def append_record(self, word_id, freq):
            '''
            Append a record to a document's words dict
            '''
            self.words_dict[word_id] = freq # key is the index of word, value if its frequency
            # self.Cj += freq

        def __repr__(self):
            return "<Document object (%s)>" % self.__str__()

        def __str__(self):
            return "Document id: (%s). Patient id: %s, Words %s, Temporal Label %s" % (
                self.doc_id, self.pat_id, len(self.words_dict), self.t_d)




def run(args):
    cmd = args.cmd
    BASE_FOLDER = args.input
    STORE_FOLDER = args.output
    print(BASE_FOLDER)
    print(STORE_FOLDER)

    if cmd == 'process':
        path = os.path.join(BASE_FOLDER, 'data.csv')
        labels = os.path.join(BASE_FOLDER, 'times.csv')
        Corpus.build_from_mixehr_fileformat(path, labels, STORE_FOLDER)

    elif cmd == 'split':
        testing_rate = args.testing_rate
        c = Corpus.read_corpus_from_directory(BASE_FOLDER)
        Corpus.split_train_test(c, testing_rate, STORE_FOLDER)

if __name__ == '__main__':
    run(parser.parse_args(['process', '-im', '-n', '150', './data/', './store/']))
    # run(parser.parse_args(['split', 'store/', 'store/']))

