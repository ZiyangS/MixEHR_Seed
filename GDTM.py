import time
import math
import numpy as np
from corpus import Corpus
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.special import gammaln

mini_val = 1e-6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GDTM(nn.Module):
    def __init__(self, corpus, seeds_topic_matrix, batch_size, out='./store/'):
        """
        Arguments:
            corpus: document class.
            seeds_topic_matrix: V x K matrix, each element represents the existence of seed word w for topic k.
            batch_size: batch size for a minibatch
            out: output path
        """
        super(GDTM, self).__init__()
        self.out = out  # folder to save experiments
        self.full_batch_generator = Corpus.generator_full_batch(corpus)
        self.C = corpus.C # C is number of words in the corpus, use for updating gamma for SCVB0
        self.M = batch_size # document number in a mini batch
        self.mini_batch_generator = Corpus.generator_mini_batch(corpus, self.M) # default batch size 1000
        self.D = corpus.D # document number in full batch
        self.K = seeds_topic_matrix.shape[1]
        self.T = corpus.T
        self.V = corpus.V  # vocabulary size of regular words
        self.seeds_topic_matrix = seeds_topic_matrix # V x K matrix, if exists value which indicates seed word w (row w) for topic k (column k)
        self.S = self.seeds_topic_matrix.sum(axis=0) # number of seed words for each topic
        self.beta = 0.1 # hyperparameter for prior of regular topic mixture phi_r
        self.beta_sum = self.beta * self.V
        self.mu = 0.1 # hyperparameter for prior of seed topic mixture phi_s
        self.mu_sum = self.mu * self.S # mu_sum is a K-length vector, mu_sum[k] is the sum of mu over all seed word (S[k]
        self.pi_init = 0.7
        self.pi = torch.full([self.K], self.pi_init, requires_grad=False, device=device) # hyperparameter weight for indicator x
        # token variables
        self.exp_m = torch.zeros(self.D, self.K, requires_grad=False, device=device)
        self.exp_n = torch.zeros(self.V, self.K, requires_grad=False, device=device)
        self.exp_s = torch.zeros(self.V, self.K, requires_grad=False, device=device) # use V to represent, regular word for a topic is 0
        self.exp_q_z = 0
        self.eta = torch.rand(self.T, self.K, requires_grad=False, device=device)
        self.alpha = self.alpha_softplus_act().to(device) # T x K
        self.initialize_tokens()
        # variational distribution for eta via amortizartion, eta is T x K matrix
        self.eta_hidden_size = 200 # number of hidden units for rnn
        self.eta_dropout = 0.0 # dropout rate on rnn for eta
        self.eta_nlayers = 3 # number of layers for eta
        self.delta = 0.01  # prior variance
        self.q_eta_map = nn.Linear(self.V, self.eta_hidden_size)
        self.q_eta = nn.LSTM(self.eta_hidden_size, self.eta_hidden_size, self.eta_nlayers, dropout=self.eta_dropout)
        self.mu_q_eta = nn.Linear(self.eta_hidden_size+self.K, self.K, bias=True)
        self.logsigma_q_eta  = nn.Linear(self.eta_hidden_size+self.K, self.K, bias=True)
        # optimizer
        self.clip = 0
        self.lr = 0.0001
        self.wdecay = 1.2e-6
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wdecay)
        self.max_logsigma_t = 5.0 # avoid the value to be too big
        self.min_logsigma_t = -5.0

    def initialize_tokens(self):
        '''
        initialize tokens E[n_wk], E[s_wk], E[m_dk]
        '''
        print("Initialize tokens")
        for i, d in enumerate(self.mini_batch_generator):  # For each epoach, we sample a series of mini_batch data once
            print("Running for %d minibatch", i)
            batch_docs, batch_indices, batch_times, batch_C = d  # batch_C is total number of words within a minibatch for SCVB0
            batch_BOW = torch.zeros(len(batch_docs), self.V, dtype=torch.int, requires_grad=False, device=device)  # M x V
            for d_i, (doc_id, doc) in enumerate(zip(batch_indices, batch_docs)):
                for word_id, freq in doc.words_dict.items():
                    batch_BOW[d_i, word_id] = freq
            for d_i, doc_id in enumerate(batch_indices):
                BOW_nonzero = torch.nonzero(batch_BOW[d_i]).squeeze(dim=1)
                self.exp_s[BOW_nonzero] += self.seeds_topic_matrix[BOW_nonzero] * batch_BOW[d_i, BOW_nonzero].unsqueeze(1) * (self.pi_init)
                self.exp_n[BOW_nonzero] += self.seeds_topic_matrix[BOW_nonzero] * batch_BOW[d_i, BOW_nonzero].unsqueeze(1) * (1-self.pi_init)
                self.exp_n[BOW_nonzero] += (1-self.seeds_topic_matrix)[BOW_nonzero] * batch_BOW[d_i, BOW_nonzero].unsqueeze(1) / (self.K - 1)
                self.exp_m[doc_id] = 1 / self.K * batch_BOW[d_i].sum()
        self.exp_m_sum = torch.sum(self.exp_m, dim=1,) # sum over k, exp_m is [D K] dimensionality, exp_m_sum is D-len vector
        self.exp_n_sum = torch.sum(self.exp_n, dim=0) # sum over w, exp_n is [V K] dimensionality, exp_n_sum is K-len vector
        self.exp_s_sum = torch.sum(self.exp_s, dim=0) # sum over w, exp_p is [V K] dimensionality, exp_s_sum is K-len vector

    def get_kl_z(self, batch_indices, batch_times):
        '''
        compute the elbo excluding eta, kl with respect to eta is computed seperately after estimation by neural network
        '''
        # compute kl(q_z || p_z), should be positive
        kl_z = 0
        # E_q[log q(z | gamma)]
        kl_z = self.exp_q_z
        # E_q[ log p(z | alpha), alpha is softplus(eta)
        p_z = gammaln(torch.sum(self.alpha[batch_times], dim=1)) - torch.sum(gammaln(self.alpha[batch_times]), dim=1) + \
                torch.sum(gammaln(self.alpha[batch_times] + self.exp_m[batch_indices]), dim=1) - \
                gammaln(torch.sum(self.alpha[batch_times], dim=1) + self.exp_m_sum[batch_indices])
        kl_z -= torch.sum(p_z)
        # E_q[log p(w | z, beta, mu, pi)], overflow
        # alpha = nn.Softplus(self.eta)
        # for k in range(self.K):
        #     log_sum_n_terms = gammaln(self.beta_sum) - self.V*gammaln(self.beta) + \
        #                 np.sum(gammaln(self.exp_n[:, k]+self.beta) + self.exp_n[:, k]*np.log(1-self.pi[k])) - \
        #                 gammaln(self.exp_n_sum[k] + self.beta_sum)
        #     log_sum_s_terms = gammaln(self.mu_sum[k]) - self.S[k]*gammaln(self.mu) + \
        #                 np.sum(gammaln(self.exp_s[:, k]+self.mu) + self.exp_s[:, k]*np.log(self.pi[k])) - \
        #                 gammaln(self.exp_s_sum[k] + self.mu_sum[k])
        # print(logsumexp(a=log_sum_n_terms, b=log_sum_s_terms))
        # kl += np.log(np.exp(log_sum_n_terms) + np.exp(log_sum_s_terms))
        return kl_z

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
        Returns KL(N(q_mu, q_logsigma) || N(p_mu, p_logsigma))
        """
        if p_mu is not None and p_logsigma is not None: # compute kl divergence for eta_{1:t-1}
            sigma_q_sq = torch.exp(q_logsigma)
            sigma_p_sq = torch.exp(p_logsigma)
            kl = (sigma_q_sq + (q_mu - p_mu)**2) / (sigma_p_sq + mini_val)
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
        return (weight.new_zeros(nlayers, 1, nhid), weight.new_zeros(nlayers, 1, nhid)) # (h0, c0), h0 and c0 are 3 x 1 x 200

    def get_eta(self, rnn_inp):
        '''
        structured amortized inference for eta
        eta is T x K matrix, each eta_t is estimated given eta{1:t-1} (i.e. eta{t-1})
        eta_t thus is modeled by neural network with concatenated input of rnn_input and eta_{t-1}
        '''
        inp = self.q_eta_map(rnn_inp).unsqueeze(1) # q_eta_map: T x V -> T x eta_hidden_size -> T x 1 x eta_hidden_size
        hidden = self.init_hidden()
        output, _ = self.q_eta(inp, hidden)
        output = output.squeeze() # output is T x eta_hidden_size
        etas = torch.zeros(self.T, self.K).to(device)
        kl_eta = []
        # get eta_0 and kl(eta_0) initially, eta_0 depends on zeros as eta_t is dependent on eta_{t-1},
        inp_0 = torch.cat([output[0], torch.zeros(self.K,).to(device)], dim=0) # we have k-len zero value vector because eta_{t-1} for eta_0 is not exist, size is eta_hidden_size (200) + topic size
        mu_0 = self.mu_q_eta(inp_0)
        logsigma_0 = self.logsigma_q_eta(inp_0)
        etas[0] = self.reparameterize(mu_0, logsigma_0)
        p_mu_0 = torch.zeros(self.K,).to(device)
        logsigma_p_0 = torch.zeros(self.K,).to(device)
        kl_0 = self.get_kl_eta(mu_0, logsigma_0, p_mu_0, logsigma_p_0) # equal to self.get_kl_eta(mu_0, logsigma_0).sum()
        kl_eta.append(kl_0)
        for t in range(1, self.T): # get eta_{1:T} given previous eta_{t-1}
            inp_t = torch.cat([output[t], etas[t-1]], dim=0)
            mu_t = self.mu_q_eta(inp_t)
            logsigma_t = self.logsigma_q_eta(inp_t)
            if (logsigma_t > self.max_logsigma_t).sum() > 0:
                logsigma_t[logsigma_t > self.max_logsigma_t] = self.max_logsigma_t
            elif (logsigma_t < self.min_logsigma_t).sum() > 0:
                logsigma_t[logsigma_t < self.min_logsigma_t] = self.min_logsigma_t
            etas[t] = self.reparameterize(mu_t, logsigma_t)
            p_mu_t = etas[t-1]
            logsigma_p_t = torch.log(self.delta * torch.ones(self.K,).to(device))
            kl_t = self.get_kl_eta(mu_t, logsigma_t, p_mu_t, logsigma_p_t)
            kl_eta.append(kl_t)
        kl_eta = torch.stack(kl_eta).sum()
        return etas, kl_eta

    def alpha_softplus_act(self):
        '''
        compute alpha using eta with softplus function
        '''
        return F.softplus(self.eta)

    def SCVB0(self, batch_BOW, batch_indices, batch_times, batch_C, iter_n):
        # temp_exp_m = torch.zeros(self.D, self.K, device=device)
        temp_exp_m_batch = torch.zeros(batch_BOW.shape[0], self.K, device=device)
        temp_exp_n = torch.zeros(self.V, self.K, device=device)
        temp_exp_s = torch.zeros(self.V, self.K, device=device)
        gamma_sr_sum = torch.zeros(self.K, device=device)
        # M step
        for d_i, doc_id in enumerate(batch_indices):
            temp_gamma_ss = torch.zeros(self.V, self.K, device=device) #  V x K  # non-seed regular word will be zero
            temp_gamma_sr = torch.zeros(self.V, self.K, device=device)
            temp_gamma_rr = torch.zeros(self.V, self.K, device=device)
            BOW_nonzero = torch.nonzero(batch_BOW[d_i]).squeeze(dim=1)
            t_d = batch_times[d_i]
            # seed word and seed topic
            temp_gamma_ss[BOW_nonzero] = self.seeds_topic_matrix[BOW_nonzero] * (self.exp_m[doc_id] + self.alpha[t_d]) \
                                         * (self.mu + self.exp_s[BOW_nonzero]) / (self.mu_sum + self.exp_s_sum) * self.pi
            # seed word but regular topic
            temp_gamma_sr[BOW_nonzero] = self.seeds_topic_matrix[BOW_nonzero] * (self.exp_m[doc_id] + self.alpha[t_d]) \
                                         * (self.beta + self.exp_n[BOW_nonzero]) / (self.beta_sum + self.exp_n_sum) * (1-self.pi)
            # regular word must be regular topic
            temp_gamma_rr[BOW_nonzero] = (1-self.seeds_topic_matrix[BOW_nonzero]) * (self.exp_m[doc_id] + self.alpha[t_d]) \
                                         * (self.beta + self.exp_n[BOW_nonzero]) / (self.beta_sum + self.exp_n_sum)
            # normalization
            temp_gamma_s_sum = temp_gamma_ss.sum(dim=1).unsqueeze(1) + temp_gamma_sr.sum(dim=1).unsqueeze(1)
            temp_gamma_r_sum = temp_gamma_rr.sum(dim=1).unsqueeze(1)
            temp_gamma_ss /= temp_gamma_s_sum + mini_val
            temp_gamma_sr /= temp_gamma_s_sum + mini_val
            temp_gamma_rr /= temp_gamma_r_sum + mini_val
            # calculate frequency * gamma for each word in a document
            seed_word_list = torch.nonzero(self.seeds_topic_matrix[BOW_nonzero].sum(axis=1)).squeeze(1)
            non_seed_word_list = torch.nonzero(1-self.seeds_topic_matrix[BOW_nonzero].sum(axis=1)).squeeze(1)
            # words can be viewed as seed words for a topic
            temp_gamma_s = (self.pi*temp_gamma_ss + (1-self.pi)*(temp_gamma_sr+temp_gamma_rr))[BOW_nonzero][seed_word_list]
            # words can not be viewed as seed words for a topic, thus it is regular words for all topics
            temp_gamma_r = temp_gamma_rr[BOW_nonzero][non_seed_word_list]
            temp_gamma = torch.cat((temp_gamma_s, temp_gamma_r), 0)
            # temp_exp_m[doc_id] += torch.sum(torch.cat((temp_gamma_s, temp_gamma_r), 0) * batch_BOW[d_i, BOW_nonzero].unsqueeze(1), dim=0)
            temp_exp_m_batch[d_i] += torch.sum(temp_gamma * batch_BOW[d_i, BOW_nonzero].unsqueeze(1), dim=0)
            temp_exp_n += (temp_gamma_sr + temp_gamma_rr) * batch_BOW[d_i].unsqueeze(1)
            temp_exp_s += temp_gamma_ss * batch_BOW[d_i].unsqueeze(1)
            gamma_sr_sum += temp_gamma_sr.sum(0)
            self.exp_q_z += torch.sum(temp_gamma * torch.log(temp_gamma+mini_val)) # used for update ELBO
        # E step
        # update expected terms
        rho = 1 / math.pow((iter_n + 5), 0.9)
        # self.exp_m[batch_indices] = (1-rho)*self.exp_m[batch_indices] + rho*temp_exp_m[batch_indices]
        self.exp_m[batch_indices] = (1-rho)*self.exp_m[batch_indices] + rho*temp_exp_m_batch
        self.exp_m_sum = torch.sum(self.exp_m, dim=1) # sum over k, exp_m is [D K] dimensionality
        self.exp_n = (1-rho)*self.exp_n + rho*temp_exp_n*self.C/batch_C
        self.exp_s = (1-rho)*self.exp_s + rho*temp_exp_s*self.C/batch_C
        self.exp_n_sum = torch.sum(self.exp_n, dim=0) # sum over w, exp_n is [V K] dimensionality
        self.exp_s_sum = torch.sum(self.exp_s, dim=0) # sum over w, exp_p is [V K] dimensionality
        self.update_hyperparams(gamma_sr_sum) # update hyperparameters

    # def pred_ind(self, index_found):
    #     pred_x = 0
    #     pred_z = index_found % self.K # get topic k from 0 to K
    #     if index_found < self.K:
    #             pred_x = 1
    #     return pred_x, pred_z
    #
    # def infer_ind(self, docs, indices):
    #     infer_x = {doc.doc_id: np.zeros(len(doc.words_dict)) for doc in docs}  # D x N_d
    #     infer_z = {doc.doc_id: np.zeros(len(doc.words_dict)) for doc in docs}  # D x N_d
    #     for d_i, doc in zip(indices, docs):
    #         for w_i, (word_id, freq) in enumerate(doc.words_dict.items()): # w_i is the index of word in each document
    #             index_found = np.argmax(np.concatenate((self.gamma_ss[doc.doc_id][w_i], self.gamma_sr[doc.doc_id][w_i], self.gamma_rr[doc.doc_id][w_i])))
    #             infer_x[doc.doc_id][w_i], infer_z[doc.doc_id][w_i] = self.pred_ind(index_found)
    #     return infer_x, infer_z
    #
    # def exp_suff(self, docs, indices):
    #     infer_x, infer_z = self.infer_ind(docs, indices)
    #     E_m_dk = np.zeros((self.D, self.K))
    #     E_s_wk = np.zeros((self.V, self.K))
    #     E_n_wk = np.zeros((self.V, self.K))
    #     for d_i, doc in zip(indices, docs):
    #         for w_i, (word_id, freq) in enumerate(doc.words_dict.items()):  # w_i is the index of word in each document
    #             E_m_dk[d_i] += (self.gamma_ss[doc.doc_id][w_i] + self.gamma_sr[doc.doc_id][w_i] + self.gamma_rr[doc.doc_id][w_i]) * freq/2
    #             # E_m_dk[d_i] += (self.pi*(self.gamma_ss[doc.doc_id][w_i]) + (1-self.pi)*(self.gamma_sr[doc.doc_id][w_i]+
    #             # self.gamma_rr[doc.doc_id][w_i]))*freq/2
    #             if infer_x[doc.doc_id][w_i] == 1: # x_indicator is assigned as 1, seed topic
    #                 assign_k = int(infer_z[doc.doc_id][w_i])
    #                 E_s_wk[w_i, assign_k] += self.gamma_ss[doc.doc_id][w_i, assign_k] * freq
    #             else: # x_indicator is 0, regular topic
    #                 for k in range(self.K):
    #                     E_n_wk[word_id, k] += (self.gamma_rr[doc.doc_id][w_i, k] + self.gamma_sr[doc.doc_id][w_i, k])*freq
    #     return E_m_dk, E_n_wk, E_s_wk
    #
    # def CVB0_generative(self, docs, indices, times, C=None):
    #     # E step
    #     for d_i, doc in zip(indices, docs):
    #         # (the number of words in a doc) x K, not use V x K to save space
    #         temp_gamma_rr = np.zeros((len(doc.words_dict), self.K))
    #         temp_gamma_sr = np.zeros((len(doc.words_dict), self.K))
    #         temp_gamma_ss = np.zeros((len(doc.words_dict), self.K))  # non-seed regular word will be zero
    #         for w_i, (word_id, freq) in enumerate(doc.words_dict.items()):  # w_i is the index of word in each document
    #             # word_id represents the index of word in vocabulary
    #             for k in range(self.K):  # update gamma_dik
    #                 if word_id in self.topic_seeds_dict[k]:  # seed word, could be regular topic or seed topic
    #                     # update seed topic
    #                     temp_gamma_ss[w_i][k] = (self.exp_m[d_i][k] + self.alpha[doc.t_d][k]) * (self.mu + self.exp_s[word_id, k]) \
    #                                             / (self.mu_sum[k] + self.exp_s_sum[k]) * self.pi[k]
    #                     # update regular topic
    #                     temp_gamma_sr[w_i][k] = (self.exp_m[d_i][k] + self.alpha[doc.t_d][k]) * (self.beta + self.exp_n[word_id, k]) \
    #                                             / (self.beta_sum + self.exp_n_sum[k]) * (1 - self.pi[k])
    #                 else:  # regular word, must be regular topic
    #                     temp_gamma_rr[w_i][k] = (self.exp_m[d_i][k] + self.alpha[doc.t_d][k]) * (self.beta + self.exp_n[word_id, k])\
    #                                             / (self.beta_sum + self.exp_n_sum[k])
    #                     # temp_gamma_rr[w_i][k] *= (self.pi[k] + self.exp_n_sum[k] + self.beta_sum) / (
    #                     #             2 * self.pi[k] + self.exp_n_sum[k] + self.beta_sum + self.exp_s_sum[k] + self.mu_sum[k])
    #             # normalization
    #             temp_gamma_s_sum = np.sum(temp_gamma_ss[w_i]) + np.sum(temp_gamma_sr[w_i])
    #             temp_gamma_r_sum = np.sum(temp_gamma_rr[w_i])
    #             temp_gamma_ss[w_i] /= temp_gamma_s_sum + mini_val
    #             temp_gamma_sr[w_i] /= temp_gamma_s_sum + mini_val
    #             temp_gamma_rr[w_i] /= temp_gamma_r_sum + mini_val
    #         self.gamma_ss[doc.doc_id] = temp_gamma_ss
    #         self.gamma_sr[doc.doc_id] = temp_gamma_sr
    #         self.gamma_rr[doc.doc_id] = temp_gamma_rr
    #     # m step
    #     # update expected terms for next iteration
    #     self.exp_m, self.exp_n, self.exp_s = self.exp_suff(docs, indices)
    #     self.exp_m_sum = np.sum(self.exp_m, axis=1) # sum over k, exp_m is [D K] dimensionality, exp_m_sum[d] should be N_d / C_j
    #     self.exp_n_sum = np.sum(self.exp_n, axis=0)  # sum over w, exp_n is [V K] dimensionality
    #     self.exp_s_sum = np.sum(self.exp_s, axis=0)  # sum over w, exp_p is [V K] dimensionality
    #     self.update_hyperparams(docs) # update hyperparameters

    def update_hyperparams(self, gamma_sr_sum):
        '''
        update hyperparameters pi using Bernoulli trial
        '''
        self.pi = self.exp_s_sum / (self.exp_s_sum + gamma_sr_sum + mini_val)
        # fill pi_init for pi_k with non-computed topic k (as gamma is 0) or very low value
        self.pi = torch.where(self.pi > 0.1, self.pi, torch.ones(self.K, device=device)*self.pi_init)

    def inference_SCVB_SGD(self, max_epoch=10, save_every=1):
        '''
        inference algorithm for dynamic seed-guided topic model, apply stochastic collaposed variational inference for latent variable z,
        and apply stochastic gradient descent for dynamic variables \eta (\alpha)
        '''
        elbo = [0,]
        max_epoch = 100
        for epoch in range(0, max_epoch):
            print("Training for epoch", epoch)
            batch_n = (self.D // self.M) + 1
            for i, d in enumerate(self.mini_batch_generator): # For each epoach, we sample a series of mini_batch data once
                print("Running for %d minibatch", i)
                batch_docs, batch_indices, batch_times, batch_C = d  # batch_C is total number of words within a minibatch for SCVB0
                batch_BOW = torch.zeros(len(batch_docs), self.V, dtype=torch.int, requires_grad=False, device=device) # document number (not M) x V
                for d_i, (doc_id, doc) in enumerate(zip(batch_indices, batch_docs)):
                    for word_id, freq in doc.words_dict.items():
                        batch_BOW[d_i, word_id] = freq
                rnn_input = self.get_rnn_input(batch_BOW, batch_times) # obtain T x V input for RNN
                start_time = time.time()
                # update eta via LSTM/Transformer model using SGD, this dynamic component can be replaced by Kalman filter
                self.optimizer.zero_grad()
                self.zero_grad()
                self.eta, kl_eta = self.get_eta(rnn_input)
                self.alpha = self.alpha_softplus_act() # alpha is T x K
                with torch.no_grad():
                    self.SCVB0(batch_BOW, batch_indices, batch_times, batch_C, epoch * batch_n + i)
                kl_z = self.get_kl_z(batch_indices, batch_times)
                loss = kl_eta + kl_z
                print(kl_eta, kl_z)
                loss.backward()
                if self.clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)
                self.optimizer.step()
                loss *= self.C/batch_C
                print("took %s seconds for minibatch %s" % (time.time() - start_time, epoch))
                elbo.append(loss.detach().cpu().numpy().item())
                print("epoch: %s, minibatch %s, elbo: %s, elbo diff: %s" % (epoch, i, elbo[-1], np.abs(elbo[-1] - elbo[-2])))
                self.exp_q_z = 0 # update to zero for next epoch
            self.save_parameters(epoch)

    def get_rnn_input(self, batch_BOW, batch_times):
        '''
        get rnn input from documents
        :param batch_BOW: BOW representation of documents for a mini batch
        :param batch_times: temporal labels of documents for a mini batch
        :return: a normalized bag-of-words representation of documents in minibatch across time labels, a T x V matrix
        '''
        batch_times = torch.from_numpy(batch_times).to(device)
        rnn_input = torch.zeros(self.T, self.V, device=device)
        cnt = torch.zeros(self.T, device=device)
        for t in range(self.T):
            tmp = (batch_times == t).nonzero()
            docs_t = batch_BOW[tmp].squeeze().sum(dim=0)
            rnn_input[t] += docs_t
            cnt[t] += len(tmp)
        rnn_input = torch.div(rnn_input, cnt.unsqueeze(1)+mini_val) # T vector to T x 1, (T x V) / (T x 1), normalize over documents
        return rnn_input

    def save_parameters(self, epoch):
        torch.save(self.exp_m, "./parameters/exp_m_%s.pt" % (epoch))
        torch.save(self.exp_n, "./parameters/exp_n_%s.pt" % (epoch))
        torch.save(self.exp_s, "./parameters/exp_s_%s.pt" % (epoch))
        torch.save(self.pi, "./parameters/pi_%s.pt" % (epoch))
        torch.save(self.eta, "./parameters/eta_%s.pt" % (epoch))
        torch.save(self.alpha, "./parameters/alpha_%s.pt" % (epoch))
