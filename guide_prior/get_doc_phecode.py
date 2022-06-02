import pandas as pd
import numpy as np
import torch
import pickle
import time
from corpus import Corpus

vocab_ids = pickle.load(open("../mapping/icd_vocab_ids.pkl", "rb"))
inv_vocab_ids = {v: k for k, v in vocab_ids.items()}

# we use GPU, printed result is "cuda"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seeds_topic_matrix = torch.load("../phecode_mapping/seed_topic_matrix.pt", map_location=device)  # get seed word-topic mapping, V x K matrix
print(seeds_topic_matrix.shape)  # 7262 as 7262 words are seed words across topics
V, K = seeds_topic_matrix.shape
print(V, K)
c = Corpus.read_corpus_from_directory('../store/train/', 'corpus.pkl')
print(c.D, c.V)
print('obtain D x K matrix')
document_phecode_matrix = torch.zeros((c.D, K), device=device)
pat_d = []
for d_i, doc in enumerate(c.dataset):
    pat_d.append(doc.doc_id)
    for v, freq in doc.words_dict[0].items():
        document_phecode_matrix[d_i] += seeds_topic_matrix[v] * freq
        # print(v, inv_vocab_ids[v], freq, torch.sum(seeds_topic_matrix[v]))
    print(doc, torch.sum(document_phecode_matrix[d_i]))
torch.save(document_phecode_matrix, "document_phecode_matrix.pt")
