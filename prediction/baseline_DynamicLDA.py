import os, time
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from corpus import Corpus
from prediction.SVM import SVM, test_SVM
from prediction.RF import RF, test_RF
from prediction.LASSO import LASSO, test_LASSO
from prediction.RidgeR import ridge, test_ridge
from prediction.LDA import classicLDA, test_classicLDA



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
theta = torch.load("../parameters/dynamicLDA_theta.pt")
theta = theta.cpu().detach().numpy()
print(theta.shape)

seeds_topic_matrix = torch.load("../phecode_mapping/seed_topic_matrix.pt")  # get seed word-topic mapping, V x K matrix
V, K = seeds_topic_matrix.shape
corpus = Corpus.read_corpus_from_directory('../test_store/')
BOW = None # get patient-level BOW representation
P = {}  # get patient-level BOW instead of document-level BOW
for d in Corpus.generator_full_batch(corpus):# get BOW representation
    docs, indices, times, C = d
    P_count = 0
    for doc in docs:
        if doc.pat_id not in P:
            P[doc.pat_id] = P_count
            P_count += 1
    BOW = np.zeros((len(P), K))  # D x K
    for d_i, (doc_id, doc) in enumerate(zip(indices, docs)):
        p_i = P[doc.pat_id]
        for word_id, freq in doc.words_dict.items():
            phecode_id = torch.nonzero(seeds_topic_matrix[word_id])
            if len(phecode_id):
                BOW[p_i, phecode_id.squeeze()] += freq
D = BOW.shape[0] # X: P x K

path = "E:/code/David_data/"
disease_fact = pd.read_csv(path + 'patients_facts.csv', index_col=0) # 1409304
select_patient = disease_fact[disease_fact.index.isin(P.keys())]
diseases = list(disease_fact.columns)
disease_n = len(disease_fact.columns)
label_raw = select_patient.to_numpy(dtype='bool')
label = np.zeros((D, 14)).astype('bool')  # y: P * 14 diseases
for pid, l in zip(P.keys(), label_raw):
    label[P[pid]] = l
ratio = label.sum(axis=0)/label.shape[0]
print(diseases)
print(ratio)

model = 'SVM'
ITER = 10
all_train_scores, all_test_scores = [], []
np.random.seed(42)
iter_seeds = np.random.randint(100, size=ITER)
for it in range(ITER):
    np.random.seed(iter_seeds[it])
    train_X, test_X, train_label, test_label = train_test_split(BOW, label, test_size=0.2) # split: 8:2

    if model == 'SVM':
        clf_SVM, train_scores = SVM(train_X, train_label, diseases)
        test_scores = test_SVM('SVM', clf_SVM, test_X, test_label, diseases)
    elif model == 'RF':
        clf_RF, train_scores = RF(train_X, train_label, diseases)
        test_scores = test_RF('Random Forest', clf_RF, test_X, test_label, diseases)
    elif model == 'LASSO':
        clf_LASSO, train_scores = LASSO(train_X, train_label.astype('double'), diseases)
        test_scores = test_LASSO('LASSO', clf_LASSO, test_X, test_label.astype('double'), diseases)
    elif model == 'ridge':
        clf_ridge, train_scores = ridge(train_X, train_label, diseases)
        test_scores = test_ridge('ridge regression', clf_ridge, test_X, test_label, diseases)
    elif model == 'LDA':
        (LDA, clf_EN), train_scores = classicLDA(train_X, train_label, diseases, n_topics=K)
        test_scores = test_classicLDA('classic LDA', LDA, clf_EN, test_X, test_label, diseases)
    all_train_scores.append(train_scores)
    all_test_scores.append(test_scores)
    # np.save(open(experiment+'_scores.npy', 'wb'), {'disease': diseases, 'ratio': ratio, 'train': all_train_scores, 'test': all_test_scores})
