# define data structure for corpus
from typing import Mapping, List, NoReturn, Set, TypeVar
import numpy as np
import pandas as pd
import pickle
import os
import logging
import sys
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
from simulation.simulation_utils import simulation_tokenize_phecode_icd

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
    def __init__(self, docs: List[D], T, V) -> NoReturn:
        logger.info("Creating corpus...")
        self.dataset = docs # a list of Document objects
        self.D = len(docs)
        self.T = T
        self.V = V
        print(self.D)
        print(self.T)
        print(self.V)

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
        docs, indixes = zip(*batch)
        # list of docs in minibatch, indexes of docs in minibatch, docs' times in minibatch, docs' total number of words in minibatch
        return list(docs), np.array(indixes), np.array([doc[0].t_d for doc in batch])

    @staticmethod
    def build_from_DTM_fileformat(data_path: str, time_path: str, store_path: str = None):
        '''
        Reads a longitudinal EHR data and return a Corpus object.
        :param data_path: data records, no header, columns are separated with spaces.
                    It contains: doc_id, pat_id, word_id, frequency, times.
        :param time_path: time data for each document.
        :param store_path: store output Corpus object.
        '''
        def __read_time__(doc_ids, labels):
            D = len(doc_ids.keys())
            t = {}
            pbar = tqdm(doc_ids.keys())
            for i, doc in enumerate(pbar):
                time = labels[labels['index'] == doc]['age_at_diagnosis'].item()
                record = doc_ids[doc]
                t[doc] = time
                pbar.set_description("%.4f  - documents(%s)" % (100 * (i + 1) / D, record))
            return t

        def __read_docs__(data, times, doc_ids, vocab_ids):
            training = {}
            num_records = data.shape[0]
            with tqdm(total=num_records) as pbar:
                for i, row in enumerate(data.iterrows()): # for each record, append to corresponding document
                    row = row[1]
                    doc_id = row['index']
                    icd = row['icd'] # str
                    freq = row['freq']
                    if doc_id not in training:
                        training[doc_id] = Corpus.Document(doc_id, times[doc_id])
                    doc = training[doc_id]
                    doc.append_record(icd, freq)
                    pbar.set_description("%.4f  - document(%s), word(%s)"
                                         % (100 * (i + 1) / num_records, doc_id, icd))
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
        phecode_ids, vocab_ids, tokenized_phecode_icd = simulation_tokenize_phecode_icd()
        # phecode_ids: key is phecode, value is the mapped index of phecode from 1 to K-1
        # vocab_ids: key is icd, value is the mapped index of icd from 1 to V-1
        doc_ids = {} # key is doc_id of original data, value is the mapped new document index from 0 to D-1
        # vocab_ids = {}  # key is icd of original data, value is the mapped new patient index from 0 to V-1
        times_ids = {}
        ids = labels['index'].unique() # get index of each document
        for i, doc_id in enumerate(ids):
            doc_ids[doc_id] = i
        # unique_words = data['icd'].unique()
        # for i, word_id in enumerate(unique_words):
        #     vocab_ids[word_id] = i

        with open('mapping/doc_ids.pkl', 'wb') as handle:
            pickle.dump(doc_ids, handle)
        with open('mapping/vocab_ids.pkl', 'wb') as handle:
            pickle.dump(vocab_ids, handle)
        with open('mapping/phecode_ids.pkl', 'wb') as handle:
            pickle.dump(phecode_ids, handle)
        with open('mapping/tokenized_phecode_icd.pkl', 'wb') as handle:
            pickle.dump(tokenized_phecode_icd, handle)
        print("finish exporting")
        # Process and read documents
        t = __read_time__(doc_ids, labels) # read temporal labels and number of wordsof documents
        dataset = __read_docs__(data, t, doc_ids, vocab_ids) # read documents
        # Set data to Corpus object
        T = len(set(t.values()))
        V = len(vocab_ids)
        corpus = Corpus([*dataset.values()], T, V)
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
        def __init__(self, doc_id, t, words_dict: dict = None):
            '''
            Create a new document.
            '''
            # may be we should have an index to show the position in corpus
            self.doc_id = doc_id # index the order of document, doc_id for train set and test set starts from 0
            self.words_dict = words_dict if words_dict is not None else {} # key: word_id, value: frequency
            self.t_d = int(t) # t_d, the temporal label for each document

        def append_record(self, word_id, freq):
            '''
            Append a record to a document's words dict
            '''
            self.words_dict[word_id] = freq # key is the index of word in vocabulary, value if its frequency

        def __repr__(self):
            return "<Document object (%s)>" % self.__str__()

        def __str__(self): # print Document object will return this string
            return "Document id: (%s). Words %s, Temporal Label %s" % (
                self.doc_id, len(self.words_dict), self.t_d)




def run(args):
    cmd = args.cmd
    BASE_FOLDER = args.input
    STORE_FOLDER = args.output
    print(BASE_FOLDER)
    print(STORE_FOLDER)
    if cmd == 'process':
        path = os.path.join(BASE_FOLDER, 'words_df.csv')
        labels = os.path.join(BASE_FOLDER, 'times_df.csv')
        Corpus.build_from_DTM_fileformat(path, labels, STORE_FOLDER)



if __name__ == '__main__':
    run(parser.parse_args(['process', '-n', '150', './simulation/', './simulation/']))

