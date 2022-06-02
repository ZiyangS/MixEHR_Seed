import pandas as pd
import numpy as np
import torch
import pickle
import time
from corpus import Corpus
from sklearn.mixture import GaussianMixture

# we use GPU, printed result is "cuda"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu" )
print(device)

seeds_topic_matrix = torch.load("../phecode_mapping/seed_topic_matrix.pt", map_location=device)  # get seed word-topic mapping, V x K matrix
topic_prior_alpha = torch.load("topic_prior_alpha.pt", map_location=device)  # get theta, D X K matrix

c = Corpus.read_corpus_from_directory('../store/train/', 'corpus.pkl')
print(c.V)
exp_n_icd = torch.zeros(c.V[0], topic_prior_alpha.shape[1], dtype=torch.double, requires_grad=False, device=device)
exp_s_icd = torch.zeros(c.V[0], topic_prior_alpha.shape[1], dtype=torch.double, requires_grad=False, device=device)
for d_i, doc in enumerate(c.dataset):
    print(d_i)
    doc_id = doc.doc_id
    for word_id, freq in doc.words_dict[0].items(): # word_index v and freq
        # update seed words
        exp_s_icd[word_id] += seeds_topic_matrix[word_id] * freq * topic_prior_alpha[d_i] * 1 # * 0.7
        exp_n_icd[word_id] += seeds_topic_matrix[word_id] * freq * topic_prior_alpha[d_i] * 1 # * 0.3
        # update regular words
        exp_n_icd[word_id] += (1-seeds_topic_matrix)[word_id] * freq * topic_prior_alpha[d_i]

print(c.V, topic_prior_alpha.shape[1])
exp_n_med = torch.zeros(c.V[1], topic_prior_alpha.shape[1], dtype=torch.double, requires_grad=False, device=device)
for d_i, doc in enumerate(c.dataset):
    print(d_i)
    doc_id = doc.doc_id
    for word_id, freq in doc.words_dict[1].items(): # word_index v and freq
        exp_n_med[word_id] += topic_prior_alpha[d_i] # if freq is 1, just add alpha prior
        # exp_n_med[word_id] += freq * topic_prior_alpha[d_i] # if freq is not 1, add freq * alpha prior

print(exp_n_icd.dtype, exp_s_icd.dtype, topic_prior_alpha.dtype, exp_n_med.dtype)

torch.save(exp_n_icd, "init_exp_n_icd.pt")
torch.save(exp_s_icd, "init_exp_s_icd.pt")
torch.save(topic_prior_alpha, "init_exp_m.pt")
torch.save(exp_n_med, "init_exp_n_med.pt")
