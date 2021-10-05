import os
import re
import time
import h5py
import numpy as np
from scipy.special import gamma, loggamma, gammaln, digamma
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
            corpus: document class. Because doc length varies with each other, it's not a D*V matrix.
            topic_seeds_dict: it indicates the seed words for each topic, all of them are tokenization.
        """
        super(GDTM, self).__init__()
        self.out = out  # folder to save experiments

        self.full_batch_generator = Corpus.generator_full_batch(corpus) # full batch size
        self.C = corpus.C # C is the number of words in the corpus, use for updating gamma for SCVB0
        self.mini_batchgenerator = Corpus.generator_mini_batch(corpus, 5) # default batch size 1000

        # self.BOW = corpus.BOW_representation()
        # self.doc_t_labels = corpus.corpus_time_labels()
        self.D = corpus.D
        self.K = K
        self.T = corpus.T  # D-length array, each value t_d represents the temporal property of document d
        self.V = corpus.V  # vocabulary size of regular words
        self.topic_seeds_dict = topic_seeds_dict # a mapping dict, key is a topic, value is associated seed words
        self.S = np.zeros(self.K, dtype=np.int) # the size of seed words for each topic
        for k, icds in enumerate(self.topic_seeds_dict.values()):
            self.S[k] = len(icds)
        self.beta = 0.1 # hyperparameter for prior of\phi_r
        self.beta_sum = self.beta * self.V
        self.mu = 0.1 # hyperparameter for prior of \phi_s
        self.mu_k = []
        self.mu_sum = np.zeros(self.K)
        for k in range(self.K):
            self.mu_k.append(np.array([self.mu] * self.S[k]))
        self.mu_sum = self.mu * self.S # mu_sum is a K-length vector as S is different for each topic k
        # we need to check original code of GLDA
        self.pi = np.full(self.K, 0.7)  # hyperparameter for prior on weight \pi

        # variational parameters, maybe we do not need to use it
        # self.gamma_r = np.zeros((self.D, self.V, self.K)) # each item is \gamma_dik
        # self.gamma_s = np.zeros((self.D, self.V, self.K))  # each item is \gamma_dik
        # self.gamma = np.zeros((self.D, self.V, self.K))
        # self.gamma_ss = np.zeros((self.D, self.V, self.K)) # each item is \gamma_dik for seed word and seed topic
        # self.gamma_sr = np.zeros((self.D, self.V, self.K)) # each item is \gamma_dik for seed word but regular topic
        # self.gamma_rr = np.zeros((self.D, self.V, self.K)) # each item is \gamma_dik for regular word which must be regular topic

        # token variables
        self.exp_m = np.random.rand(self.D, self.K)
        self.exp_n = np.random.rand(self.V, self.K)
        self.exp_s = np.random.rand(self.V, self.K) # use V to represent, regular word will be 0

        self.eta = torch.rand(self.T, self.K)
        self.alpha = self.alpha_softplus_act(self.eta, comp_met="init")
        self.alpha = self.alpha.detach().cpu().numpy() # T x K

        # variational distribution for eta via amortizartion, eta is T x K matrix
        self.eta_hidden_size = 200 # number of hidden units for rnn
        self.eta_dropout = 0.0 # dropout rate on rnn for eta
        self.eta_nlayers = 3 # number of layers for eta
        self.delta = 0.005  # prior variance
        self.q_eta_map = nn.Linear(self.V, self.eta_hidden_size)
        self.q_eta = nn.LSTM(self.eta_hidden_size, self.eta_hidden_size, self.eta_nlayers, dropout=self.eta_dropout)
        self.mu_q_eta = nn.Linear(self.eta_hidden_size+self.K, self.K, bias=True)
        self.logsigma_q_eta  = nn.Linear(self.eta_hidden_size+self.K, self.K, bias=True)
        # optimizer
        self.lr = 0.005
        self.wdecay = 1.2e-6
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wdecay)
        # save model parameters
        # self.tune_model_parameter =  ['pi', 'gamma', 'exp_m', 'exp_n', 'exp_s']
        # self.parameters = ['mu', 'beta', 'pi', 'gamma', 'exp_m', 'exp_n', 'exp_s']
        self.initialize_tokens()

    def initialize_tokens(self):
        for i, d in enumerate(self.full_batch_generator):
            batch_docs, batch_indices, batch_times, batch_Cj = d # batch_Cj is total number of minibatch
            for d_i, doc in zip(batch_indices, batch_docs):
                for word_id, freq in doc.words_dict.items(): # word_id represents the index of word in vocabulary
                    for k in range(self.K):
                        if word_id in self.topic_seeds_dict[k]:
                            self.exp_n[word_id][k] = (1-self.pi[k]) * 1
                            self.exp_s[word_id][k] = self.pi[k] * 1
                        else:
                            self.exp_n[word_id][k] = 1 / (self.K-1)
                for k in range(self.K): # todo: it can be simplied by using np.full
                    self.exp_m[d_i][k] = 1 / self.K * doc.Cj
            self.exp_m_sum = np.sum(self.exp_m, axis=1) # sum over k, exp_m is [D K] dimensionality, exp_m_sum is D-len vector
            self.exp_n_sum = np.sum(self.exp_n, axis=0) # sum over w, exp_n is [V K] dimensionality, exp_n_sum is K-len vector
            self.exp_s_sum = np.sum(self.exp_s, axis=0) # sum over w, exp_p is [V K] dimensionality, exp_s_sum is K-len vector
            self.gamma_ss = {doc.doc_id: np.zeros((len(doc.words_dict), self.K)) for doc in batch_docs} # D x N_d x K
            self.gamma_sr = {doc.doc_id: np.zeros((len(doc.words_dict), self.K)) for doc in batch_docs} # D x N_d x K
            self.gamma_rr = {doc.doc_id: np.zeros((len(doc.words_dict), self.K)) for doc in batch_docs} # D x N_d x K
            for d_i, doc in zip(batch_indices, batch_docs):
                # here we need to have a mapping for test, for real implement, we remove it
                for w_i, (word_id, freq) in enumerate(doc.words_dict.items()): # w_i represents the index of word in documents
                    for k in range(self.K):
                        if word_id in self.topic_seeds_dict[k]:
                            self.gamma_ss[doc.doc_id][w_i][k] += (self.exp_m[d_i][k] + self.alpha[doc.t_d][k]) \
                                * (self.exp_s[word_id][k] + self.mu) / (self.exp_s_sum[k] + self.mu_sum[k]) * self.pi[k]
                            self.gamma_sr[doc.doc_id][w_i][k] += (self.exp_m[d_i][k] + self.alpha[doc.t_d][k]) \
                                * (self.exp_n[word_id][k] + self.beta) / (self.exp_n_sum[k] + self.beta_sum) * (1 - self.pi[k])
                        else:
                            self.gamma_rr[doc.doc_id][w_i][k] += (self.exp_m[d_i][k] + self.alpha[doc.t_d][k]) \
                                * (self.exp_n[word_id][k] + self.beta) / (self.exp_n_sum[k] + self.beta_sum)


    def compute_elbo(self):
        '''
        compute the elbo excluding eta, kl with respect to eta is computed seperately after estimation by neural network
        '''
        eps = np.finfo(float).eps
        elbo = 0
        # # E_q[log p(z | alpha)], self.alpha[self.doc_t_labels] is a D x K matrix
        # elbo_p_z = gammaln(np.sum(self.alpha[self.doc_t_labels], axis=1)) - np.sum(gammaln(self.alpha[self.doc_t_labels]), axis=1) + \
        #         np.sum(gammaln(self.alpha[self.doc_t_labels] + self.exp_m), axis=1) - \
        #         gammaln(np.sum(self.alpha[self.doc_t_labels] + self.exp_m, axis=1))
        # elbo += np.sum(elbo_p_z)
        # # E_q[log p(w | z, beta, mu, pi)], check overflow issue
        # # here is problem, what is S
        # elbo_w_z = [gamma(self.beta_sum) / np.power(gamma(self.beta), self.V) \
        #             * np.prod(gamma(self.exp_n[:, k]+self.beta) * np.power(1-self.pi[k], self.exp_n[:, k])) \
        #             / gamma(self.exp_n_sum[k] + self.beta_sum) + \
        #             gamma(self.mu_sum) / np.power(gamma(self.beta), self.S) \
        #             * np.prod(gamma(self.exp_s[:, k] + self.mu) * np.power(self.pi[k], self.exp_s[:, k])) \
        #             / gamma(self.exp_s_sum[k] + self.mu_sum)
        #             for k in range(self.K)]
        # elbo += np.log(elbo_w_z)
        # # - E_q[log q(z | gamma)]
        # elbo -= self.exp_q_z
        return elbo

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
        eta is T x K matrix, each eta_t is estimated given eta{1:t-1} (i.e. eta{t-1})
        eta_t thus is modeled by neural network with concatenated input of rnn_input and eta_{t-1}
        '''
        print(rnn_inp.get_device())
        inp = self.q_eta_map(rnn_inp).unsqueeze(1) # q_eta_map: T x V -> T x eta_hidden_size -> T x 1 x eta_hidden_size
        hidden = self.init_hidden()
        output, _ = self.q_eta(inp, hidden)
        output = output.squeeze() # output is T x eta_hidden_size

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
            inp_t = torch.cat([output[t], etas[t-1]], dim=0)
            mu_t = self.mu_q_eta(inp_t)
            logsigma_t = self.logsigma_q_eta(inp_t)
            etas[t] = self.reparameterize(mu_t, logsigma_t)
            p_mu_t = etas[t-1]
            logsigma_p_t = torch.log(self.delta * torch.ones(self.K,).to(device))
            kl_t = self.get_kl_eta(mu_t, logsigma_t, p_mu_t, logsigma_p_t)
            kl_eta.append(kl_t)
        kl_eta = torch.stack(kl_eta).sum()
        return etas, kl_eta

    def forward(self, rnn_input):
        '''
        receives input Tensors and produces output Tensors
        '''
        print('forward automatically running?')
        self.etas, kl_eta = self.get_eta(rnn_input)  # change to rnn_inp
        return kl_eta

    def eta_exp(self, eta):
        '''
        compute the expected terms related to eta
        '''
        # self.alpha = np.log(1 + np.exp(eta))
        pass

    def alpha_softplus_act(self, eta, comp_met="infer"):
        '''
        compute alpha using eta with softplus function
        '''
        print('test alpha')
        print(eta)
        # print(eta.item())
        if comp_met == "infer":
            print("infer")
            self.alpha = nn.Softplus(eta)
            print(self.alpha)
        elif comp_met == "init":
            print("init")
            print(F.softplus(eta))
            return F.softplus(eta)

    def CVB0(self, docs, indices, times, Cj):
        temp_exp_m = np.zeros((self.D, self.K))
        temp_exp_n = np.zeros((self.V, self.K))
        temp_exp_s = np.zeros((self.V, self.K))
        # E step
        for d_i, doc in zip(indices, docs):
            # (the number of words in a doc) x K, not use V x K to save space
            temp_gamma_rr = np.zeros((len(doc.words_dict), self.K))
            temp_gamma_sr = np.zeros((len(doc.words_dict), self.K))
            temp_gamma_ss = np.zeros((len(doc.words_dict), self.K)) # non-seed regular word will be zero
            for w_i, (word_id, freq) in enumerate(doc.words_dict.items()): # w_i is the index of word in each document
                # print(w_i, word_id)
                # word_id represents the index of word in vocabulary
                for k in range(self.K): # update gamma_dik
                    # todo: we could update K topics at same time? because we need to check [z_di = k] it is difficult
                    if word_id in self.topic_seeds_dict[k]: # seed word, could be regular topic or seed topic
                        # update seed topic
                        temp_gamma_ss[w_i][k] = (self.exp_m[d_i][k] + self.alpha[doc.t_d][k]) * (self.mu + self.exp_s[word_id, k]) \
                                                / (self.mu_sum[k] + self.exp_s_sum[k]) * self.pi[k]
                        # update regular topic
                        temp_gamma_sr[w_i][k] = (self.exp_m[d_i][k] + self.alpha[doc.t_d][k]) * (self.beta + self.exp_n[word_id, k]) \
                                                / (self.beta_sum + self.exp_n_sum[k]) * (1 - self.pi[k])
                    else:  # regular word, must be regular topic
                        temp_gamma_rr[w_i][k] = (self.exp_m[d_i][k] + self.alpha[doc.t_d][k]) * (self.beta + self.exp_n[word_id, k])\
                                                / (self.beta_sum + self.exp_n_sum[k])
                        # temp_gamma_rr[w_i][k] *= (self.pi[k] + self.exp_n_sum[k] + self.beta_sum) / (
                        #             2 * self.pi[k] + self.exp_n_sum[k] + self.beta_sum + self.exp_s_sum[k] + self.mu_sum[k])
                # normalization
                temp_gamma_s_sum = np.sum(temp_gamma_ss[w_i]) + np.sum(temp_gamma_sr[w_i])
                temp_gamma_r_sum = np.sum(temp_gamma_rr[w_i])
                temp_gamma_ss[w_i] /= temp_gamma_s_sum
                temp_gamma_sr[w_i] /= temp_gamma_s_sum
                temp_gamma_rr[w_i] /= temp_gamma_r_sum
                # calculate frequency * gamma for each word, iterate words in documents
                # temp_exp_m[d_i] += (self.pi*temp_gamma_ss[w_i] + (1-self.pi)*(temp_gamma_rr[w_i] + temp_gamma_sr[w_i])) * freq / 2
                temp_exp_m[d_i] += (temp_gamma_ss[w_i] + temp_gamma_rr[w_i] + temp_gamma_sr[w_i]) * freq / 2
                temp_exp_n[w_i] += (temp_gamma_rr[w_i] + temp_gamma_sr[w_i]) * freq
                temp_exp_s[w_i] += temp_gamma_ss[w_i] * freq
            self.gamma_ss[doc.doc_id] = temp_gamma_ss
            self.gamma_sr[doc.doc_id] = temp_gamma_sr
            self.gamma_rr[doc.doc_id] = temp_gamma_rr
        # m step
        # update expected terms for next iteration
        self.exp_m = temp_exp_m
        self.exp_n = temp_exp_n
        self.exp_s = temp_exp_s
        self.exp_m_sum = np.sum(self.exp_m, axis=1) # sum over k, exp_m is [D K] dimensionality, exp_m_sum[d] should be N_d / C_j
        self.exp_n_sum = np.sum(self.exp_n, axis=0) # sum over w, exp_n is [V K] dimensionality
        self.exp_s_sum = np.sum(self.exp_s, axis=0) # sum over w, exp_p is [V K] dimensionality
        # self.update_hyperparams(docs) # update hyperparameters
        elbo_z = self.compute_elbo()
        return elbo_z

    def pred_ind(self, index_found):
        pred_x = 0
        pred_z = index_found % self.K # get topic k from 0 to K
        if index_found < self.K:
                pred_x = 1
        return pred_x, pred_z

    def infer_ind(self, docs, indices):
        infer_x = {doc.doc_id: np.zeros(len(doc.words_dict)) for doc in docs}  # D x N_d
        infer_z = {doc.doc_id: np.zeros(len(doc.words_dict)) for doc in docs}  # D x N_d
        for d_i, doc in zip(indices, docs):
            for w_i, (word_id, freq) in enumerate(doc.words_dict.items()): # w_i is the index of word in each document
                index_found = np.argmax(np.concatenate((self.gamma_ss[doc.doc_id][w_i], self.gamma_sr[doc.doc_id][w_i], self.gamma_rr[doc.doc_id][w_i])))
                infer_x[doc.doc_id][w_i], infer_z[doc.doc_id][w_i] = self.pred_ind(index_found)
        return infer_x, infer_z

    def exp_suff(self, docs, indices):
        infer_x, infer_z = self.infer_ind(docs, indices)
        E_m_dk = np.zeros((self.D, self.K))
        E_s_wk = np.zeros((self.V, self.K))
        E_n_wk = np.zeros((self.V, self.K))
        for d_i, doc in zip(indices, docs):
            for w_i, (word_id, freq) in enumerate(doc.words_dict.items()):  # w_i is the index of word in each document
                E_m_dk[d_i] += (self.gamma_ss[doc.doc_id][w_i] + self.gamma_sr[doc.doc_id][w_i] + self.gamma_rr[doc.doc_id][w_i]) * freq/2
                # E_m_dk[d_i] += (self.pi*(self.gamma_ss[doc.doc_id][w_i]) + (1-self.pi)*(self.gamma_sr[doc.doc_id][w_i]+
                # self.gamma_rr[doc.doc_id][w_i]))*freq/2
                if infer_x[doc.doc_id][w_i] == 1: # x_indicator is assigned as 1, seed topic
                    assign_k = int(infer_z[doc.doc_id][w_i])
                    E_s_wk[w_i, assign_k] += self.gamma_ss[doc.doc_id][w_i, assign_k] * freq
                else: # x_indicator is 0, regular topic
                    for k in range(self.K):
                        E_n_wk[word_id, k] += (self.gamma_rr[doc.doc_id][w_i, k] + self.gamma_sr[doc.doc_id][w_i, k])*freq
        return E_m_dk, E_n_wk, E_s_wk

    def CVB0_generative(self, docs, indices, times, Cj):
        # E step
        for d_i, doc in zip(indices, docs):
            # (the number of words in a doc) x K, not use V x K to save space
            temp_gamma_rr = np.zeros((len(doc.words_dict), self.K))
            temp_gamma_sr = np.zeros((len(doc.words_dict), self.K))
            temp_gamma_ss = np.zeros((len(doc.words_dict), self.K))  # non-seed regular word will be zero
            for w_i, (word_id, freq) in enumerate(doc.words_dict.items()):  # w_i is the index of word in each document
                # word_id represents the index of word in vocabulary
                for k in range(self.K):  # update gamma_dik
                    # todo: we could update K topics at same time? because we need to check [z_di = k] it is difficult
                    if word_id in self.topic_seeds_dict[k]:  # seed word, could be regular topic or seed topic
                        # update seed topic
                        temp_gamma_ss[w_i][k] = (self.exp_m[d_i][k] + self.alpha[doc.t_d][k]) * (self.mu + self.exp_s[word_id, k]) \
                                                / (self.mu_sum[k] + self.exp_s_sum[k]) * self.pi[k]
                        # update regular topic
                        temp_gamma_sr[w_i][k] = (self.exp_m[d_i][k] + self.alpha[doc.t_d][k]) * (self.beta + self.exp_n[word_id, k]) \
                                                / (self.beta_sum + self.exp_n_sum[k]) * (1 - self.pi[k])
                    else:  # regular word, must be regular topic
                        temp_gamma_rr[w_i][k] = (self.exp_m[d_i][k] + self.alpha[doc.t_d][k]) * (self.beta + self.exp_n[word_id, k])\
                                                / (self.beta_sum + self.exp_n_sum[k])
                        # temp_gamma_rr[w_i][k] *= (self.pi[k] + self.exp_n_sum[k] + self.beta_sum) / (
                        #             2 * self.pi[k] + self.exp_n_sum[k] + self.beta_sum + self.exp_s_sum[k] + self.mu_sum[k])
                # normalization
                temp_gamma_s_sum = np.sum(temp_gamma_ss[w_i]) + np.sum(temp_gamma_sr[w_i])
                temp_gamma_r_sum = np.sum(temp_gamma_rr[w_i])
                temp_gamma_ss[w_i] /= temp_gamma_s_sum
                temp_gamma_sr[w_i] /= temp_gamma_s_sum
                temp_gamma_rr[w_i] /= temp_gamma_r_sum
            self.gamma_ss[doc.doc_id] = temp_gamma_ss
            self.gamma_sr[doc.doc_id] = temp_gamma_sr
            self.gamma_rr[doc.doc_id] = temp_gamma_rr
        # m step
        # update expected terms for next iteration
        self.exp_m, self.exp_n, self.exp_s = self.exp_suff(docs, indices)
        self.exp_m_sum = np.sum(self.exp_m, axis=1) # sum over k, exp_m is [D K] dimensionality, exp_m_sum[d] should be N_d / C_j
        self.exp_n_sum = np.sum(self.exp_n, axis=0)  # sum over w, exp_n is [V K] dimensionality
        self.exp_s_sum = np.sum(self.exp_s, axis=0)  # sum over w, exp_p is [V K] dimensionality
        # self.update_hyperparams(docs) # update hyperparameters
        elbo_z = self.compute_elbo()
        return elbo_z

    def update_hyperparams(self, docs):
        '''
        update hyperparameters \pi, this is not useful as gamma_r_sum is extremely large
        '''
        eps = np.finfo(float).eps
        gamma_s_sum = np.zeros(self.K)
        gamma_r_sum = np.zeros(self.K)
        for doc in docs:
            # todo: gamma should be changed to sparse matrix to speed up
            gamma_s_sum += self.gamma_ss[doc.doc_id].sum(axis=0) # sum over word to obtain K-len vector
            gamma_r_sum += self.gamma_sr[doc.doc_id].sum(axis=0) + self.gamma_rr[doc.doc_id].sum(axis=0)
        self.pi = gamma_s_sum / (gamma_r_sum + gamma_s_sum + eps)

    def inference_SCVB_AEVI(self, args, max_epoch=1000, save_every=100):
        elbo = [0,]
        for epoch in range(0, max_epoch):
            print("Training for epoch", epoch)
            # for testing, we use full batch generator
            for i, d in enumerate(self.full_batch_generator): # for each epoach, we sample mini_batch data once
                print("Running for %d minibatch", i)
                batch_docs, batch_indices, batch_times, batch_Cj = d  # batch_Cj is total number of minibatch
                rnn_input = self.get_rnn_input(batch_docs, batch_indices, batch_times) # get T x V input for rnn
                # todo: maybe rnn_input should be normalized as DETM paper
                start_time = time.time()
                # elbo_expect_eta = self.CVB0(batch_docs, batch_indices, batch_times, batch_Cj)
                elbo_expect_eta = self.CVB0_generative(batch_docs, batch_indices, batch_times, batch_Cj)
                # update eta via LSTM/Transformer network by gradient descent, this component can be replaced by Kalman filter
                # self.optimizer.zero_grad()
                # self.zero_grad()
                # self.eta, kl_eta = self.get_eta(rnn_input) # ingest rnn_input
                # self.alpha = self.alpha_softplus_act(self.eta, comp_met="infer")
                # self.alpha = self.alpha.detach().cpu().numpy()  # T x K
                # loss = kl_eta + elbo_expect_eta
                # loss.backward()
                # if self.clip > 0:
                #     torch.nn.utils.clip_grad_norm_(self.parameters(), args.clip)
                self.optimizer.step()
                print("\t took %s seconds" % (time.time() - start_time))
                # elbo.append(loss.detach().cpu().numpy())
                # print("epoch %s: elbo %s, diff %s "% (epoch, elbo[-1], np.abs(elbo[-1] - elbo[-2])))
                # if epoch % save_every == 0:
                #     self.save_model(epoch)

    def get_rnn_input(self, batch_docs, batch_indices, batch_times) :
        # rnn_input = torch.zeros(self.T, self.V).to(device)
        rnn_input = torch.zeros(self.T, self.V)
        # cnt = torch.zeros(self.T).to(device)
        cnt = torch.zeros(self.T)
        batch_size = len(batch_docs)
        BOW_docs = np.zeros((batch_size, self.V)) # batch_D x V
        for d_i, doc in zip(batch_indices, batch_docs): # d_i is the index of document in corpus
            words_dict = doc.words_dict
            for word_id, freq in words_dict.items():
                BOW_docs[d_i, word_id] = freq # update BOW[d, w] = freq
        for t in range(self.T):
            tmp = (batch_times == t).nonzero()
            docs_t = BOW_docs[tmp].squeeze().sum(0) # check here
            rnn_input[t] += docs_t
            cnt[t] += len(tmp)
        rnn_input = rnn_input / cnt.unsqueeze(1) # T vector to T x 1, (T x V) / (T x 1), normalize
        return rnn_input.to(device) # place on GPU

    # def save_model(self, iter):
    #     with h5py.File(os.path.join(self.out, 'model_gdtm_k%s_iter%s.hdf5' % (self.K, iter)), 'w') as hf:
    #         for param in self.parameters:
    #             if param == 'gamma':
    #                 pass
    #             else:
    #                 hf.create_dataset(param, data=self.__getattribute__(param))



