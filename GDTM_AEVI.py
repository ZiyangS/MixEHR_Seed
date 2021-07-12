import os
import re
import time
import h5py
import numpy as np
from scipy.special import digamma, gammaln
from scipy.stats import norm
import pickle
from corpus import Corpus

class GDTM():
    def __init__(self, K, corpus:Corpus, out='./store/'):
        """
        Arguments:
            K: Number of topics
            corpus: document class. Because doc length varies with each other, it's not a D*M matrix.
        """
        self.out = out  # folder to save experiments

        self.C = corpus.C
        self.generator = Corpus.generator_full_batch(corpus)
        self.D = corpus.D
        self.K = K
        self.T = corpus.T  # D-length array, each value t_d represents the temporal property of document d
        self.V = corpus.V  # vocabulary size for regular words
        self.S = corpus.S  # vocabulary size for seed words

        self.mu = 0.5 # hyperparameter for prior on topic proportion of regular topics \phi_r
        self.beta = 0.5 # hyperparameter for prior on topic proportion of seed topics \phi_s
        # we need to check original code of GLDA
        self.pi = np.random.normal(0.7, 0, self.K)  # hyperparameter for prior on weight \pi

        # self.mu_sum = np.sum(self.mu)  # scalar value
        # self.beta_sum = np.sum(self.beta)  # scalar value

        # variational parameters
        self.gamma = np.zeros((self.D, self.V, self.K)) # each item is \gamma_dik
        self.exp_z_avg = np.zeros((self.D, self.K))
        # token variables
        self.exp_m = np.random.rand((self.D, self.K))
        self.exp_n = np.random.rand((self.V, self.K))
        self.exp_s = np.random.rand((self.S, self.K))
        for d in range(self.D):
            self.exp_m[d] /= np.sum(self.exp_m[d])
        for w in range(self.V):
            self.exp_n[w] /= np.sum(self.exp_n[w])
        for w in range(self.S):
            self.exp_s[w] /= np.sum(self.exp_s[w])
        self.exp_m_sum = np.sum(self.exp_m, axis=1) # sum over k, exp_m is [D K] dimensionality
        self.exp_n_sum = np.sum(self.exp_n, axis=0) # sum over w, exp_n is [V K] dimensionality
        self.exp_s_sum = np.sum(self.exp_s, axis=0) # sum over w, exp_p is [S K] dimensionality

        # Model parameters
        self.tune_model_parameter =  ['pi', 'gamma', 'exp_m', 'exp_n', 'exp_s']
        self.parameters = ['mu', 'beta', 'pi', 'gamma', 'exp_m', 'exp_n', 'exp_s']

    def ELBO(self):
        eps = np.finfo(float).eps
        ELBO = None
        # E_q[log p(alpha)]
        # E_q[log p(z | alpha)] -
        # E_q[log p(w | z, beta, mu, pi)]
        # - E_q[log q(z | gamma)]
        # - E_q[log q(alpha)]
        return ELBO

    def CVB0(self, doc):
        temp_exp_m = np.zeros((self.D, self.K))
        temp_exp_n = np.zeros((self.V, self.K))
        temp_exp_s = np.zeros((self.S, self.K))
        # E step
        for pat in doc:
            temporal_doc = pat.td
            temp_gamma = np.zeros((len(pat.words_dict), self.K)) # W X K
            for i, counts in enumerate(pat.words_dict.items()): # i is the index of word in a document
                (w_ind, w_i), freq = counts # w_i represents the index of word in vocabulary
                # we update K topics once
                temp_gamma[i] = (self.alpha[temporal_doc] + self.exp_m[pat.patient_id])
                # todo:  we should minus -ij, here
                if w_ind == 0: # regular word, it must be regular topic
                    temp_gamma[i] *= (self.beta + self.exp_n[w_i-1]) / (self.beta_sum + self.exp_n_sum)
                else: # seed word
                    pass
                    # todo: how to distinguish regular topic or seed topic???
                temp_gamma[i] /= np.sum(temp_gamma[i])
                temp_exp_m[pat.patient_id] += temp_gamma[i] * pat.word_freq[w_i]
                if w_ind == 0: # regular word, update n
                    temp_exp_n[w_i] += temp_gamma[i] * freq
                else:          # seed word, update s
                    temp_exp_s[w_i] += temp_gamma[i] * freq
            self.gamma[pat.patient_id] = temp_gamma
            self.exp_n[pat.patient_id] = temp_exp_n[pat.patient_id]
            self.exp_z_avg[pat.patient_id] = temp_exp_m[pat.patient_id] / pat.pat.total_counts
        # m step
        self.exp_m = temp_exp_m
        self.exp_n = temp_exp_n
        self.exp_s = temp_exp_s
        self.exp_m_sum = np.sum(self.exp_m, axis=1) # sum over k, exp_m is [D K] dimensionality
        self.exp_n_sum = np.sum(self.exp_n, axis=0) # sum over w, exp_n is [V K] dimensionality
        self.exp_s_sum = np.sum(self.exp_s, axis=0) # sum over w, exp_p is [S K] dimensionality
        self.update_hyperparams() # update hyperparameters
        # how to update alpha?

    def inference_svb(self, max_iter=500, save_every=100):
        elbo = [0, ]
        iter = 1
        while iter <= max_iter:
            for i, d in enumerate(self.generator):
                batch_patient, batch_i, M = d
                start_time = time.time()
                self.CVB0(batch_patient)
                print("\t took %s seconds" % (time.time() - start_time))
                elbo.append(self.ELBO())
                print("%s elbo %s diff %s "%(iter , elbo[-1], np.abs(elbo[-1] - elbo[-2])))
                if iter % save_every == 0:
                    self.save_model(iter)
                iter += 1
                if not iter <= max_iter:
                    break

    def update_hyperparams(self):
        '''
        update hyperparameters \pi and others
        '''
        pass

    def save_model(self, iter):
        with h5py.File(os.path.join(self.out, 'model_gdtm_k%s_iter%s.hdf5' % (self.K, iter)), 'w') as hf:
            for param in self.parameters:
                if param == 'gamma':
                    pass
                else:
                    hf.create_dataset(param, data=self.__getattribute__(param))



if __name__ == '__main__':
    c_train = Corpus.read_corpus_from_directory("../dataset/cv1/train")
    c_test = Corpus.read_corpus_from_directory("../dataset/cv1/test")
    K = 50
    gdtm = GDTM(K, c_train)
    # gdtm.inference_svb(max_iter=500, save_every=100)
