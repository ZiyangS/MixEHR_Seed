import os
import re
import time
import h5py
import numpy as np
from scipy.special import digamma, gammaln
from scipy.stats import norm
import pickle
from corpus import Corpus
import math
import torch
from torch import nn, optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GDTM(nn.Module):
    def __init__(self, K, corpus:Corpus, topic_seeds_dict: dict, out='./store/'):
        """
        Arguments:
            K: Number of topics
            corpus: document class. Because doc length varies with each other, it's not a D*V matrix.?
            actually it could be D X V matrix, each document represents as a count in vacob
            topic_seeds_dict: it indicates the seed words for each topic
        """
        super(GDTM, self).__init__()
        self.out = out  # folder to save experiments

        # self.C = corpus.C
        self.generator = Corpus.generator_mini_batch(corpus, 1000) # could change batch size
        # self.BOW = corpus.BOW_representation()
        # self.doc_t_labels = corpus.corpus_time_labels()
        self.D = corpus.D
        self.K = K
        self.T = corpus.T  # D-length array, each value t_d represents the temporal property of document d
        self.V = corpus.V  # vocabulary size of regular words
        self.topic_seeds_dict = topic_seeds_dict # a mapping dict, key is a topic, value is associated seed words
        self.S = [0] * self.K # the size of seed words for each topic
        for k, key in enumerate(self.topic_seeds_dict):
            self.S[k] = len(self.topic_seeds_dict[key])

        self.eta = np.random.rand(self.T, self.K)
        self.alpha_act = nn.Softplus() # Softplus (smoothReLu)
        self.alpha = self.alpha_act(self.eta)

        self.beta = 0.5 # hyperparameter for prior of\phi_r
        self.beta_sum = self.beta * self.V
        self.mu = 0.5 # hyperparameter for prior of \phi_s
        self.mu_sum = self.mu * self.S # mu_sum is a K-length vector as S is different for each topic k
        # we need to check original code of GLDA
        # self.pi = np.random.normal(0.7, 0, self.K)  # hyperparameter for prior on weight \pi
        self.pi = np.array(0.7, self.K)  # initialized weights \pi

        # variational parameters
        self.gamma_r = np.zeros((self.D, self.V, self.K)) # each item is \gamma_dik
        self.gamma_s = np.zeros((self.D, self.V, self.K))  # each item is \gamma_dik
        self.gamma = np.zeros((self.D, self.V, self.K))
        # token variables
        self.exp_m = np.random.rand(self.D, self.K)
        self.exp_n = np.random.rand(self.V, self.K)
        # self.exp_s = []
        # for k in range(self.K):
        #     self.exp_s[k] = np.random.rand(len(self.S[k])) # K X S[k]
        self.exp_s = np.random.rand(self.V, self.K) # maybe use the full vocab to calculate, non-seed regular word will be 0
        # normalize
        for d in range(self.D):
            self.exp_m[d] /= np.sum(self.exp_m[d])
        for w in range(self.V):
            self.exp_n[w] /= np.sum(self.exp_n[w])
        for w in range(self.V):
            self.exp_s[w] /= np.sum(self.exp_s[w])
        # self.exp_s_sum = []
        # for k, s_k in enumerate(self.S):
        #     self.exp_s_sum[k] =
        self.exp_m_sum = np.sum(self.exp_m, axis=1) # sum over k, exp_m is [D K] dimensionality, exp_m_sum is D-len vector
        self.exp_n_sum = np.sum(self.exp_n, axis=0) # sum over w, exp_n is [V K] dimensionality, exp_n_sum is K-len vector
        self.exp_s_sum = np.sum(self.exp_s, axis=0) # sum over w, exp_p is [V K] dimensionality, exp_s_sum is K-len vector
        # for k, s_k in enumerate(self.S):
        #     self.exp_s_sum[k] =

        # variational distribution for eta via amortizartion
        # eta is T x K matrix, each eta[t] is K-len vector prior for \theta after softplus function
        self.t_hidden_size = 800  # dimension of hidden space of q(theta)
        self.eta_hidden_size = 200 # number of hidden units for rnn
        self.rho_size = 300 # dimension of rho
        self.emb_size =300 #dimension of embeddings
        self.enc_drop = 0.0 # dropout rate on encoder
        self.eta_nlayers = 3 # number of layers for eta
        self.t_drop = nn.Dropout(self.enc_drop) # dropout Layers
        self.delta = 0.005  # prior variance
        # self.train_embeddings = args.train_embeddings

        self.q_eta_map = nn.Linear(self.V, self.eta_hidden_size)
        self.q_eta = nn.LSTM(self.eta_hidden_size, self.eta_hidden_size, self.eta_nlayers, dropout=self.enc_drop)
        self.mu_q_eta = nn.Linear(self.eta_hidden_size+self.K, self.K, bias=True)
        self.logsigma_q_eta  = nn.Linear(self.eta_hidden_size+self.K, self.K, bias=True)

        # optimizer
        self.lr = 0.005
        self.wdecay = 1.2e-6
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wdecay)

        # Model parameters
        # self.tune_model_parameter =  ['pi', 'gamma', 'exp_m', 'exp_n', 'exp_s']
        # self.parameters = ['mu', 'beta', 'pi', 'gamma', 'exp_m', 'exp_n', 'exp_s']

    def compute_elbo(self):
        '''
        compute the elbo excluding eta, kl with respect to eta is computed seperately after estimation by neural network
        '''
        eps = np.finfo(float).eps
        elbo_z = None
        # E_q[log p(z | alpha)] -
        # E_q[log p(w | z, beta, mu, pi)]
        # - E_q[log q(z | gamma)]
        return elbo_z

    def reparameterize(self, mu, logvar):
        """
        Returns a sample from a Gaussian distribution via reparameterization
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def get_kl_eta(self, q_mu, q_logsigma, p_mu=None, p_logsigma=None):
        """
        Returns KL( N(q_mu, q_logsigma) || N(p_mu, p_logsigma) ).
        """
        if p_mu is not None and p_logsigma is not None: # compute kl divergence for eta_{1:t-1}
            sigma_q_sq = torch.exp(q_logsigma)
            sigma_p_sq = torch.exp(p_logsigma)
            kl = (sigma_q_sq + (q_mu - p_mu)**2 ) / ( sigma_p_sq + 1e-6 )
            kl = kl - 1 + p_logsigma - q_logsigma
            kl = 0.5 * torch.sum(kl, dim=-1)
        else: # compute kl divergence for eta_0
            kl = -0.5 * torch.sum(1 + q_logsigma - q_mu.pow(2) - q_logsigma.exp(), dim=-1)
        return kl

    def init_hidden(self):
        """
        Initializes the first hidden state of the RNN used as inference network for \eta.
        """
        weight = next(self.parameters())
        nlayers = self.eta_nlayers
        nhid = self.eta_hidden_size
        return (weight.new_zeros(nlayers, 1, nhid), weight.new_zeros(nlayers, 1, nhid)) # (h0, c0)

    def get_eta(self, rnn_inp):
        '''
        structured amortized inference for eta
        eta is T x K matrix, each eta_t is estimated given eta{1:t-1} (eta{t-1})
        eta_t thus is modeled by neural network with concatenated input of rnn_input and eta_{t-1}
        '''
        inp = self.q_eta_map(rnn_inp).unsqueeze(1) # q_eta_map: T x V -> T x eta_hidden_size -> T x 1 x eta_hidden_size
        hidden = self.init_hidden() #
        output, _ = self.q_eta(inp, hidden)
        output = output.squeeze() # output is T x eta_hidden_size
        # why we need to compute output not just used rnn_input

        etas = torch.zeros(self.T, self.K).to(device)
        kl_eta = []
        # get eta_0 and kl(eta_0) initially as eta_t is dependent on eta_{t-1}
        inp_0 = torch.cat([output[0], torch.zeros(self.K,).to(device)], dim=0) # zero K-len vector is non-exist eta_{t-1} for eta_0
        mu_0 = self.mu_q_eta(inp_0)
        logsigma_0 = self.logsigma_q_eta(inp_0)
        etas[0] = self.reparameterize(mu_0, logsigma_0)
        p_mu_0 = torch.zeros(self.K,).to(device)
        logsigma_p_0 = torch.zeros(self.K,).to(device)
        kl_0 = self.get_kl_eta(mu_0, logsigma_0, p_mu_0, logsigma_p_0)
        kl_eta.append(kl_0)
        # get eta_{1:T} given initial eta_0
        for t in range(1, self.T):
            inp_t = torch.cat([output[t], etas[t-1]], dim=0) # normalized_input(why we need to use output?) + eta_{t-1}
            mu_t = self.mu_q_eta(inp_t)
            logsigma_t = self.logsigma_q_eta(inp_t)
            etas[t] = self.reparameterize(mu_t, logsigma_t)
            p_mu_t = etas[t-1]
            logsigma_p_t = torch.log(self.delta * torch.ones(self.K,).to(device))
            kl_t = self.get_kl_eta(mu_t, logsigma_t, p_mu_t, logsigma_p_t)
            kl_eta.append(kl_t)
        kl_eta = torch.stack(kl_eta).sum()
        return etas, kl_eta

    def eta_exp(self, eta):
        '''
        compute the expected terms related to eta
        '''
        self.alpha = self.alpha_act(eta)

    def CVB0(self, docs):
        temp_exp_m = np.zeros((self.D, self.K))
        temp_exp_n = np.zeros((self.V, self.K))
        temp_exp_s = np.zeros((self.V, self.K))
        # E step
        for doc in docs: # original representation
            t_d = doc.t
            # (the number of words in a doc) x K, not use V x K to save space
            temp_gamma_rr = np.zeros((len(doc.words_dict), self.K))
            temp_gamma_sr = np.zeros((len(doc.words_dict), self.K))
            temp_gamma_ss = np.zeros((len(doc.words_dict), self.K)) # non-seed regular word will be zero
            for w_idx, counts in enumerate(doc.words_dict.items()): # w_idx is the index of word in each document
                w_i, freq = counts # w_i represents the index of word in vocabulary
                for k in range(self.K): # update gamma_dik
                    # todo: we could update K topics at same time? because we need to check [z_di = k] it is difficult
                    if w_i not in self.topic_seeds_dict[k]:   # regular word, must be regular topic
                        temp_gamma_rr[w_idx, k] = (self.alpha[t_d, k] + self.exp_m[doc.doc_id, k])\
                                                  * (self.beta + self.exp_n[w_i, k]) / (self.beta_sum + self.exp_n_sum[k])
                    else: # seed word, could be regular topic or seed topic
                        # update regular topic
                        temp_gamma_sr[w_idx, k] = (self.alpha[t_d, k] + self.exp_m[doc.doc_id, k]) *\
                                                  (self.beta + self.exp_n[w_i, k]) / (self.beta_sum + self.exp_n_sum[k])\
                                                  * (1 - self.pi[k])
                        # update seed topic
                        temp_gamma_ss[w_idx, k] = (self.alpha[t_d, k] + self.exp_m[doc.doc_id, k])\
                                                  * (self.mu + self.exp_s[w_i, k]) / (self.mu_sum[k] + self.exp_s_sum[k])\
                                                  * self.pi[k]
                temp_gamma_rr[w_idx] /= np.sum(temp_gamma_rr[w_idx])  # normalization over K topics
                temp_gamma_sr[w_idx] /= (np.sum(temp_gamma_sr[w_idx]) + np.sum(temp_gamma_ss[w_idx]))
                temp_gamma_ss[w_idx] /= (np.sum(temp_gamma_sr[w_idx]) + np.sum(temp_gamma_ss[w_idx]))
                # calculate frequency * gamma for each word, iterate words in documents
                temp_exp_m[doc.doc_id] += (temp_gamma_rr[w_idx] + temp_gamma_sr[w_idx] + temp_gamma_ss[w_idx]) * freq
                temp_exp_n[w_i] += (temp_gamma_rr[w_idx] + temp_gamma_sr[w_idx]) * freq
                temp_exp_s[w_i] += temp_gamma_ss[w_idx] * freq
            self.gamma_r[doc.doc_id] = temp_gamma_rr + temp_gamma_sr # not sure useful or not, change finally
            self.gamma_s[doc.doc_id] = temp_gamma_ss # not sure useful or not, change finally

        # m step
        # update expected terms for next iteration
        self.exp_m = temp_exp_m
        self.exp_n = temp_exp_n
        self.exp_s = temp_exp_s
        self.exp_m_sum = np.sum(self.exp_m, axis=1) # sum over k, exp_m is [D K] dimensionality
        self.exp_n_sum = np.sum(self.exp_n, axis=0) # sum over w, exp_n is [V K] dimensionality
        self.exp_s_sum = np.sum(self.exp_s, axis=0) # sum over w, exp_p is [V K] dimensionality
        # for k in range(K):
        #     self.gamma[:,:,k] = (1-self.pi[k]) * self.gamma_r[:,:,k] + self.pi[k] * self.gamma_s[:,:,k]
        self.update_hyperparams() # update hyperparameters
        elbo_z = self.compute_elbo()
        return elbo_z

    def update_hyperparams(self):
        '''
        update hyperparameters \pi
        '''
        gamma_s_sum = np.zeros(self.K)
        gamma_r_sum = np.zeros(self.K)
        for d in range(self.D):
            gamma_s_sum += self.gamma_s[d].sum(axis=0) # k vector
            gamma_r_sum += self.gamma_r[d].sum(axis=0)
        self.pi = gamma_s_sum / (gamma_r_sum + gamma_s_sum)

    def inference_svb(self, args, max_epoch=1000, save_every=100):
        elbo = [0,]
        for epoch in range(0, max_epoch):
            print("Training for epoch", epoch)
            for i, d in enumerate(self.generator): # for each epoach, we sample mini_batch data once
                print("Running for", i)
                batch_docs, batch_indices, batch_times = d # check the data type
                rnn_input = self.get_rnn_input(batch_docs, batch_times) # get input for LSTM/transformer
                # maybe rnn_input should be normalized as DETM paper
                start_time = time.time()
                elbo_z = self.CVB0(batch_docs)
                # update eta via LSTM/Transformer network by gradient descent
                self.optimizer.zero_grad()
                self.zero_grad()
                self.eta, kl_eta = self.get_eta(rnn_input)  # change to rnn_inp
                self.eta_exp(self.eta)
                loss = kl_eta + elbo_z # here is a problem, maybe only kl_eta need to be differentiated
                loss.backward()
                if self.clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), args.clip)
                self.optimizer.step()
                print("\t took %s seconds" % (time.time() - start_time))
                elbo.append(loss)
                print("epoch %s: elbo %s, diff %s "% (epoch, elbo[-1], np.abs(elbo[-1] - elbo[-2])))
                if epoch % save_every == 0:
                    self.save_model(epoch)

    def get_rnn_input(self, docs, times):
        rnn_input = torch.zeros(self.T, self.V).to(device)
        cnt = torch.zeros(self.T).to(device)
        batch_size = len(docs)
        BOW_docs = np.zeros(batch_size, self.V) # batch_D x V
        for doc in docs:
            words_dict = doc.words_dict
            for word_id, freq in words_dict.items():
                BOW_docs[doc.doc_id, word_id] = freq # update BOW[d, w] = freq
        for t in range(self.T):
            tmp = (times == t).nonzero()
            docs_t = BOW_docs[tmp].squeeze().sum(0) # check here
            rnn_input[t] += docs_t
            cnt[t] += len(tmp)
        rnn_input = rnn_input / cnt.unsqueeze(1) # T vector to T x 1, (T x V) / (T x 1), normalize
        return rnn_input

    # def save_model(self, iter):
    #     with h5py.File(os.path.join(self.out, 'model_gdtm_k%s_iter%s.hdf5' % (self.K, iter)), 'w') as hf:
    #         for param in self.parameters:
    #             if param == 'gamma':
    #                 pass
    #             else:
    #                 hf.create_dataset(param, data=self.__getattribute__(param))


# if __name__ == '__main__':
#     import json
#     with open('./phecode_mapping/phecode3_icd_dict.json') as f:
#         topic_seeds_dict = json.load(f) # we should map word by BOWs
#     gdtm = GDTM(100, './corpus', topic_seeds_dict, './result/')
#     gdtm.inference_svb(max_iter=1000, save_every=50)

