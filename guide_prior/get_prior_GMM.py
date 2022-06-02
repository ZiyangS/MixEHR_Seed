import pandas as pd
import numpy as np
import torch
import pickle
import time
from corpus import Corpus
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.linear_model import LogisticRegression

# we use GPU, printed result is "cuda"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# read phecode - ICD code mapping
seeds_topic_matrix = torch.load("../phecode_mapping/seed_topic_matrix.pt", map_location=device)  # get seed word-topic mapping, V x K matrix
V, K = seeds_topic_matrix.shape

# read document's phecode counts
document_phecode_matrix = torch.load("document_phecode_matrix.pt", map_location=device)  # get documet phecode count, D x K matrix
document_phecode_matrix = document_phecode_matrix.cpu().detach().numpy()
D, K = document_phecode_matrix.shape
print(document_phecode_matrix.shape)  # 7262 as 7262 words are seed words across topics
# doc_num_phecode = document_phecode_matrix.sum(axis=0)
# print(document_phecode_matrix[0])

# define alpha as the D X K matrix as prior
alpha_prior = np.zeros((document_phecode_matrix.shape))
for k in range(K):
    print(k)
    x = document_phecode_matrix[:, k ]
    check_sample = x[0]
    x = x.reshape(-1, 1)
    model = BayesianGaussianMixture(n_components=2, init_params='random', random_state=0).fit(x)
    # model = GaussianMixture(n_components=2, init_params='random', random_state=0).fit(x)
    mixture_proba = model.predict_proba(x)
    if (check_sample == 1 and mixture_proba[0][0] > 0.5) or (check_sample == 0 and mixture_proba[0][0] < 0.5):
        alpha_prior[:, k] = model.predict_proba(x)[:, 0]
    elif (check_sample == 1 and mixture_proba[0][1] > 0.5) or (check_sample == 0 and mixture_proba[0][1] < 0.5):
        alpha_prior[:, k] = model.predict_proba(x)[:, 1]

# save alpha prior as tensor object
alpha_prior = alpha_prior / alpha_prior.sum(axis=1, keepdims=1) + 1e-5 # normalization over K for each document
alpha_prior = torch.tensor(alpha_prior)
torch.save(alpha_prior, "topic_prior_alpha.pt")

