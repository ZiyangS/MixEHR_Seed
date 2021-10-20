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
from utils import tokenize_phecode_icd

D = TypeVar('D')

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
    def __init__(self, docs: List[D], T, V, C) -> NoReturn:
        logger.info("Creating corpus...")
        self.dataset = docs # a list of Document objects
        self.D = len(docs)
        self.T = T
        self.V = V
        self.C = C # C is the number of words in the corpus

    def __len__(self):
        return self.D

    def __getitem__(self, index):
        '''
        Generate one sample for dataLoader
        '''
        doc_sample = self.dataset[index]
        words_dict = doc_sample.words_dict
        word_freq = {}
        for word, freq in words_dict.items():
            if word not in word_freq:
                word_freq[word] = 0
            word_freq[word] += freq
        doc_sample.word_freq = word_freq
        return doc_sample, index

    # def BOW_representation(self):
    #     # full batch, we assume
    #     BOW = np.zeros(self.D, self.V)
    #     for doc in self.dataset:
    #         words_dict = doc.words_dict
    #         for word_id, freq in words_dict.items():
    #             BOW[doc.doc_id, word_id] = freq # doc_id, here is wrong
    #     return BOW

    def corpus_time_labels(self):
        # full batch, we assume
        t_labels = np.zeros(self.D)
        for doc in self.dataset:
            t_labels[doc.doc_id] = doc.t_d
        return t_labels

    @staticmethod
    def __collate_model__(batch):
        '''
        Returns a batch for each iteration of the DataLoader
        '''
        docs, indixes = zip(*batch)
        # list of docs in minibatch, indexes of docs in minibatch, docs' times in minibatch, docs' total number of words in minibatch
        return list(docs), np.array(indixes), np.array([doc[0].t_d for doc in batch]), np.sum([p[0].Cj for p in batch])

    @staticmethod
    def build_from_DTM_fileformat(data_path: str, time_path: str, store_path: str = None):
        '''
        Reads a longitudinal EHR data and return a Corpus object.
        :param data_path: data records, no header, columns are separated with spaces.
                    It contains: doc_id, pat_id, word_id, frequency, times.
        :param time_path: time data for each document.
        :param store_path: store output Corpus object.
        '''
        def __read_time_Cj__(doc_ids, labels):
            D = len(doc_ids.keys())
            t = {}
            Cj = {}
            pbar = tqdm(doc_ids.keys())
            for i, doc in enumerate(pbar):
                time = labels[labels['doc_id'] == doc]['age_at_diagnosis'].item()
                count = labels[labels['doc_id'] == doc]['Cj'].item()
                record = doc_ids[doc]
                t[doc] = time
                Cj[doc] = count
                pbar.set_description("%.4f  - documents(%s)" % (100 * (i + 1) / D, record))
            return t, Cj

        def __read_docs__(data, times, doc_ids, pat_ids, vocab_ids, Cj):
            training = {}
            num_records = data.shape[0]
            with tqdm(total=num_records) as pbar:
                for i, row in enumerate(data.iterrows()): # for each record, append to corresponding document
                    row = row[1]
                    doc_id = row['doc_id']
                    pat_id = row['pat_id']
                    icd = row['icd'] # str
                    if len(icd) == 3:
                        icd = icd[0:3] + '.' + '0'
                    else:
                        icd = icd[0:3] + '.' + icd[3]
                    # todo: here do more test to see whehter it is done, particularlly test when len(icd) =3
                    if row['icd'] in vocab_ids:
                        word_id = vocab_ids[row['icd']] # should we set here, map vocabulary to another, i think so
                    elif icd in vocab_ids:
                        word_id = vocab_ids[icd]  # should we set here, map vocabulary to another, i think so
                    else:
                        continue # as we just continue, so the pbar will not be 100% sometimes
                    freq = row['freq']
                    if doc_id not in training:
                        training[doc_id] = Corpus.Document(doc_id, pat_id, times[doc_id], Cj[doc_id])
                    doc = training[doc_id]
                    doc.append_record(word_id, freq)
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
        data = pd.read_csv(data_path) # read documents data, the file format should be .csv
        labels = pd.read_csv(time_path) # read time of documents, the file format should be .csv
        print(data)
        print(labels)
        print(labels.age_at_diagnosis.unique())
        C = data.freq.to_numpy().sum() # the number of words of a corpus
        phecode_ids, vocab_ids, tokenized_phecode_icd = tokenize_phecode_icd()
        # phecode_ids: key is phecode, value is the mapped index of phecode from 1 to K-1
        # vocab_ids: key is icd, value is the mapped index of icd from 1 to V-1
        doc_ids = {} # key is doc_id of original data, value is the mapped new document index from 0 to D-1
        pat_ids = {} # key is pat_id of original data, value is the mapped new patient index from 0 to P-1
        # vocab_ids = {}  # key is icd of original data, value is the mapped new patient index from 0 to V-1
        times_ids = {}
        ids = labels['doc_id'].unique() # get index of each document
        for i, doc_id in enumerate(ids):
            doc_ids[doc_id] = i
        # do we need to set a patient's record? after preprocessing data, we need to check
        unique_patients = labels['pat_id'].unique()
        for i, pat_id in enumerate(unique_patients):
            pat_ids[pat_id] = i
        # unique_words = data['icd'].unique()
        # for i, word_id in enumerate(unique_words):
        #     vocab_ids[word_id] = i

        with open('mapping/doc_ids.pkl', 'wb') as handle:
            pickle.dump(doc_ids, handle)
        with open('mapping/pat_ids.pkl', 'wb') as handle:
            pickle.dump(pat_ids, handle)
        with open('mapping/vocab_ids.pkl', 'wb') as handle:
            pickle.dump(vocab_ids, handle)
        with open('mapping/phecode_ids.pkl', 'wb') as handle:
            pickle.dump(phecode_ids, handle)
        with open('mapping/tokenized_phecode_icd.pkl', 'wb') as handle:
            pickle.dump(tokenized_phecode_icd, handle)
        print("finish exporting")
        # Process and read documents
        t, Cj = __read_time_Cj__(doc_ids, labels) # read temporal labels and number of wordsof documents

        dataset = __read_docs__(data, t, doc_ids, pat_ids, vocab_ids, Cj) # read documents
        # Set data to Corpus object
        T = len(set(t.values()))
        V = len(vocab_ids)
        corpus = Corpus([*dataset.values()], T, V, C)
        logger.info(f'''
        ========= DataSet Information =========
        Documents: {len(corpus.dataset)}
        Patients: {len(corpus.dataset)} # this is not right
        Times: {corpus.T}
        Word Tokes: {corpus.V}
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
            C = 0
            index = 0 # set index to zero for train set, doc_id from 0 to D-1 ，original code is -1, need to check
            dbar = tqdm(corpus)
            for doc, _, in dbar:
                dbar.set_description("Processing document %s (Patient index: %s)" % (doc.doc_id, doc.pat_id)) # check description
                doc.doc_id = index # doc_id from 0 to D-1 # here is not right
                index += 1
                C += doc.Cj
                documents.append(doc)
                if index == size and not splitted:
                    corpus_list[0] = Corpus(documents, corpus.T, corpus.W, C) # obtain train set
                    index = 0 # set index to zero for test set
                    documents = [] # initialize to store test documents
                    splitted = True
            corpus_list[1] = Corpus(documents, corpus.T, corpus.W, C) # obtain test set
            return tuple(corpus_list)

        train_size = corpus.D - int(split_rate * corpus.D)
        train, test = __split__(train_size, corpus)
        # store data
        __store_data__(os.path.join(toStore, 'train'), train)
        __store_data__(os.path.join(toStore, 'test'), test)
        logger.info("Training size: %s\nTesting size: %s\n" % (train_size, corpus.D - train_size))

    @staticmethod
    def read_corpus_from_directory(path):
        '''
        Reads existed data
        :param path: folder containing corpus files
        :return: Corpus object and metadata (patient ids, data type ids, vocab ids)
        '''
        corpus_file = os.path.join(path, "corpus.pkl")
        corpus = pickle.load(open(corpus_file, "rb"))
        docs = corpus.dataset
        # precompute all labels
        t_labels = np.zeros(len(corpus))
        # for doc in docs:
        #     # todo: change the mapped index later instead of enumerate
        #     print(doc.doc_id)
        for i, doc in enumerate(docs):
            # t_labels[doc.doc_id] = doc.t_d
            t_labels[i] = doc.t_d
        corpus.labels = t_labels
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
        def __init__(self, doc_id, pat_id, t, Cj, words_dict: dict = None):
            '''
            Create a new document.
            '''
            # may be we should have an index to show the position in corpus
            self.doc_id = doc_id # index the order of document, doc_id for train set and test set starts from 0
            self.pat_id = pat_id # index patient, it is a part of original data
            self.words_dict = words_dict if words_dict is not None else {} # key: word_id, value: frequency
            self.t_d = int(t) # t_d, the temporal label for each document
            # self.Cj = Cj # Cj is the number of words of a document
            self.Cj = 0 # Cj is the number of words of a document


        def append_record(self, word_id, freq):
            '''
            Append a record to a document's words dict
            '''
            self.words_dict[word_id] = freq # key is the index of word in vocabulary, value if its frequency
            self.Cj += freq # add could of words for this document

        def __repr__(self):
            return "<Document object (%s)>" % self.__str__()

        def __str__(self): # print Document object will return this string
            return "Document id: (%s). Patient id: %s, Words %s, Count %s, Temporal Label %s" % (
                self.doc_id, self.pat_id, len(self.words_dict), self.Cj, self.t_d)




def run(args):
    cmd = args.cmd
    BASE_FOLDER = args.input
    STORE_FOLDER = args.output
    print(BASE_FOLDER)
    print(STORE_FOLDER)

    if cmd == 'process':
        path = os.path.join(BASE_FOLDER, 'document_part_data.csv')
        labels = os.path.join(BASE_FOLDER, 'label_part_data.csv')
        Corpus.build_from_DTM_fileformat(path, labels, STORE_FOLDER)

    elif cmd == 'split':
        testing_rate = args.testing_rate
        c = Corpus.read_corpus_from_directory(BASE_FOLDER)
        Corpus.split_train_test(c, testing_rate, STORE_FOLDER)

if __name__ == '__main__':
    run(parser.parse_args(['process', '-n', '150', './data/', './store/']))
    # run(parser.parse_args(['split', 'store/', 'store/']))
