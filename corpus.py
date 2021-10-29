# define data structure for corpus
import numpy as np
import pandas as pd
import pickle
import os
import logging
import sys
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
from utils import tokenize_phecode_icd_corpus

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
    def __init__(self, docs, T, V, C):
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

    @staticmethod
    def __collate_model__(batch):
        '''
        Returns a batch for each iteration of the DataLoader
        '''
        docs, indixes = zip(*batch) # tuple
        # list of docs in minibatch, indexes of docs in minibatch, docs' times in minibatch, docs' total number of words in minibatch
        return list(docs), np.array(indixes), np.array([doc[0].t_d for doc in batch]), np.sum([p[0].Cd for p in batch])

    @staticmethod
    def build_from_GDTM_fileformat(data_path, time_path, store_path=None):
        '''
        Reads a longitudinal EHR data and return a Corpus object.
        :param data_path: data records, no header, columns are separated with spaces.
                    It contains: doc_id, pat_id, word_id, frequency, times.
        :param time_path: time data for each document.
        :param store_path: store output Corpus object.
        '''
        def __read_time_Cd__(labels):
            ids = labels['doc_id'].unique()  # get index of each document
            D = len(ids)
            t = {}
            Cd = {}
            pbar = tqdm(ids)
            for i, doc_id in enumerate(pbar):
                time = labels[labels['doc_id'] == doc_id]['age_at_diagnosis'].item()
                count = labels[labels['doc_id'] == doc_id]['Cd'].item()
                record = doc_id
                t[doc_id] = time
                Cd[doc_id] = count
                pbar.set_description("%.4f  - documents(%s)" % (100 * (i + 1) / D, record))
            return t, Cd

        def __read_docs__(data, times, vocab_ids, Cd):
            training = {}
            num_records = data.shape[0]
            with tqdm(total=num_records) as pbar:
                for i, row in enumerate(data.iterrows()): # for each record, append to its document
                    row = row[1]
                    doc_id = row['doc_id']
                    pat_id = row['pat_id']
                    word_id = vocab_ids[row['icd']]
                    freq = row['freq']
                    if doc_id not in training:
                        training[doc_id] = Corpus.Document(doc_id, pat_id, times[doc_id], Cd[doc_id])
                    doc = training[doc_id]
                    doc.append_record(word_id, freq)
                    pbar.set_description("%.4f  - document(%s), patient(%s), word(%s)" % (100 * (i + 1) / num_records, doc_id, pat_id, word_id))
                    pbar.update(1)
            return training

        def __store_data__(toStore, corpus):
            if not os.path.exists(toStore):
                os.makedirs(toStore)
            corpus_file = os.path.join(toStore, "corpus.pkl")
            logger.info("Saving: \n\t%s" % (corpus_file))
            pickle.dump(corpus, open(corpus_file, "wb"))
            logger.info("Data stored in %s" % toStore)

        data = pd.read_csv(data_path) # read documents data, the file format should be .csv
        labels = pd.read_csv(time_path) # read time of documents, the file format should be .csv
        print(data)
        print(labels)
        C = data.freq.to_numpy().sum() # number of words of a corpus
        phecode_ids, vocab_ids, tokenized_phecode_icd = tokenize_phecode_icd_corpus(list(data.icd.unique()))
        # phecode_ids: key is phecode, value is the mapped index of phecode from 1 to K-1, K is 1569
        # vocab_ids: key is icd, value is the mapped index of icd from 1 to V-1, V is 8539
        # tokenized_phecode_icd is dict {mapped phecode: [mapped ICD codes]}, len(key) is 1569, len(values) is 5741, other are regular words
        with open('mapping/vocab_ids.pkl', 'wb') as handle:
            pickle.dump(vocab_ids, handle)
        with open('mapping/phecode_ids.pkl', 'wb') as handle:
            pickle.dump(phecode_ids, handle)
        with open('mapping/tokenized_phecode_icd.pkl', 'wb') as handle:
            pickle.dump(tokenized_phecode_icd, handle)
        print("finish exporting mapping")
        # Process and read documents
        t, Cd = __read_time_Cd__(labels) # read temporal labels and number of words of documents
        dataset = __read_docs__(data, t, vocab_ids, Cd) # read documents
        T = len(set(t.values()))
        V = len(vocab_ids)
        corpus = Corpus([*dataset.values()], T, V, C) # Set data to Corpus object
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
    def split_train_test(corpus, split_rate, toStore):
        '''
        train-test split for corpus object
        '''
        assert split_rate >= .0 and split_rate <= 1., "specify the rate for splitting training and test. e.g 0.8 = 80% for testing"

        def __store_data__(toStore, corpus):
            if not os.path.exists(toStore):
                os.makedirs(toStore)
            corpus_file = os.path.join(toStore, "corpus.pkl")
            logger.info("Saving: \n\t%s" % (corpus_file))
            pickle.dump(corpus, open(corpus_file, "wb"))
            logger.info("Data stored in %s" % toStore)

        def __split__(train_size, corpus):
            documents = [] # initialize to store train documents
            corpus_list = [None, None]
            splitted = False
            C = 0
            index = 0 # set index to zero for train set, doc_id from 0 to D-1 ，original code is -1, need to check
            dbar = tqdm(corpus)
            for doc, _, in dbar:
                dbar.set_description("Processing document %s (Patient index: %s)" % (doc.doc_id, doc.pat_id)) # check description
                doc.doc_id = index # doc_id from 0 to D-1
                index += 1
                C += doc.Cd
                documents.append(doc)
                if index == train_size and not splitted:
                    corpus_list[0] = Corpus(documents, corpus.T, corpus.V, C) # obtain train set
                    index = 0 # set index to zero for test set
                    documents = [] # initialize to store test documents
                    splitted = True
            corpus_list[1] = Corpus(documents, corpus.T, corpus.V, C) # obtain test set
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
        '''
        corpus_file = os.path.join(path, "corpus.pkl")
        corpus = pickle.load(open(corpus_file, "rb"))
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
        def __init__(self, doc_id, pat_id, t, Cd, words_dict: dict = None):
            '''
            Create a new document.
            '''
            self.doc_id = doc_id # index of document, doc_id for train set and test set starts from 0
            self.pat_id = pat_id # index of patient from original data
            self.words_dict = words_dict if words_dict is not None else {} # key: word_id, value: frequency
            self.t_d = int(t) # temporal label for each document
            self.Cd = 0 # number of words of a document

        def append_record(self, word_id, freq):
            '''
            Append a record to a document's words dict
            '''
            self.words_dict[word_id] = freq # key is index of word in vocabulary, value if its frequency
            self.Cd += freq # add freq to Cd

        def __repr__(self):
            return "<Document object (%s)>" % self.__str__()

        def __str__(self): # print Document object will return this string
            return "Document id: (%s). Patient id: %s, Words %s, Count %s, Temporal Label %s" % (
                self.doc_id, self.pat_id, len(self.words_dict), self.Cd, self.t_d)


def run(args):
    cmd = args.cmd
    BASE_FOLDER = args.input
    STORE_FOLDER = args.output
    print(BASE_FOLDER)
    print(STORE_FOLDER)

    if cmd == 'process':
        path = os.path.join(BASE_FOLDER, 'document_full_data.csv')
        labels = os.path.join(BASE_FOLDER, 'label_full_data.csv')
        Corpus.build_from_GDTM_fileformat(path, labels, STORE_FOLDER)

    elif cmd == 'split':
        testing_rate = args.testing_rate
        c = Corpus.read_corpus_from_directory(BASE_FOLDER)
        Corpus.split_train_test(c, testing_rate, STORE_FOLDER)

if __name__ == '__main__':
    run(parser.parse_args(['process', '-n', '150', './data/', './store/']))
    # run(parser.parse_args(['split', 'store/', 'store/']))
