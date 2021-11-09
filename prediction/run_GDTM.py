import pandas as pd
import numpy as np
import torch
import pickle
import time
from sklearn.model_selection import train_test_split
from corpus import Corpus
from prediction.SVM import SVM, test_SVM
from prediction.RF import RF, test_RF
from prediction.LASSO import LASSO, test_LASSO
from prediction.RidgeR import ridge, test_ridge
from prediction.LDA import classicLDA, test_classicLDA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
exp_m = torch.load("../parameters/exp_m_0.pt")
D, K = exp_m.shape
alpha = torch.load("../parameters/alpha_0.pt") # T x K
c = Corpus.read_corpus_from_directory('../test_store/')
t_d = [] # D-len vector, alpha: T x K -> alpha[t_d] D x K
pat_d = []
for doc in c.dataset:
    t_d.append(doc.t_d)
    pat_d.append(doc.pat_id)
theta = (alpha[t_d] + exp_m) / (alpha[t_d].sum(dim=1).unsqueeze(1) + exp_m.sum(dim=1).unsqueeze(1))
theta = theta.cpu().detach().numpy()
print(theta)
corpus = Corpus.read_corpus_from_directory('../test_store/')
P = {}  # get patient-level BOW instead of document-level BOW
P_d = {}
P_weight = {}
theta_P = None # get patient level theta
for d in Corpus.generator_full_batch(corpus):# get BOW representation
    docs, indices, times, C = d
    P_count = 0
    for doc in docs:
        if doc.pat_id not in P:
            P[doc.pat_id] = P_count
            P_d[doc.pat_id] = [doc.doc_id]
            P_weight[doc.pat_id] = [doc.Cd]
            P_count += 1
        else:
            P_d[doc.pat_id].append(doc.doc_id)
            P_weight[doc.pat_id].append(doc.Cd)
    theta_P = np.zeros((len(P), K))  # P x V
    for d_i, (doc_id, doc) in enumerate(zip(indices, docs)):
        p_i = P[doc.pat_id]
        p_d = P_d[doc.pat_id].index(doc_id)
        theta_d = theta[d_i]
        theta_d_weight = P_weight[doc.pat_id][p_d] / np.sum(P_weight[doc.pat_id])
        theta_P[p_i] += theta_d * theta_d_weight

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

# get phecodes for diseases
phecode_ids = pickle.load(open("../mapping/phecode_ids.pkl", "rb"))
adhd_phecode = phecode_ids[313]
ami_phecdoe = phecode_ids[411]
asthma_phecdoe = phecode_ids[495]
autism_phecdoe = phecode_ids[313]
chf_phecdoe = phecode_ids[428]
COPD_phecode = phecode_ids[496]
diabetes_phecode = phecode_ids[250]
epilepsy_phecode = phecode_ids[345]
hiv_phecode = phecode_ids[71]
hypertension_phecode = phecode_ids[401]
ihd_phecode = phecode_ids[411]
schizophrenia_phecode = phecode_ids[295]
# get learned probability for patients given phecodes
theta_adhd = theta[:, adhd_phecode]
theta_ami = theta[:, ami_phecdoe]
theta_asthma = theta[:, asthma_phecdoe]
theta_autism = theta[:, autism_phecdoe]
theta_chf = theta[:, chf_phecdoe]
theta_COPD = theta[:, COPD_phecode]
theta_diabetes = theta[:, diabetes_phecode]
theta_epilepsy = theta[:, epilepsy_phecode]
theta_hiv= theta[:, hiv_phecode]
theta_hypertension = theta[:, hypertension_phecode]
theta_ihd = theta[:, ihd_phecode]
theta_schizophrenia = theta[:, schizophrenia_phecode]


model = 'SVM'
ITER = 10
all_train_scores, all_test_scores = [], []
np.random.seed(42)
iter_seeds = np.random.randint(100, size=ITER)
for it in range(ITER):
    for theta in [theta_adhd, theta_ami, theta_asthma, theta_autism, theta_chf, theta_COPD, theta_diabetes, theta_epilepsy,
        theta_hiv, theta_hypertension, theta_ihd, theta_schizophrenia]:
        np.random.seed(iter_seeds[it])
        train_X, test_X, train_label, test_label = train_test_split(theta.reshape(-1, 1), label, test_size=0.2) # split: 8:2

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

