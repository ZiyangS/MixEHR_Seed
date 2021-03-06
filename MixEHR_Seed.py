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

class MixEHR_Seed(nn.Module):
    def __init__(self, corpus, seeds_topic_matrix, modality_num, guided_modality=0, batch_size=1000, out='./store/'):
        """
        Arguments:
            corpus: document class.
            seeds_topic_matrix: V x K matrix, each element represents the existence of seed word w for topic k.
            batch_size: batch size for a minibatch
            out: output path
        """
        super(MixEHR_Seed, self).__init__()
        self.modaltiy_num = modality_num # the number of modaltiy M
        self.guided_modality = guided_modality # the guided modality
        self.out = out  # folder to save experiments

        self.full_batch_generator = Corpus.generator_full_batch(corpus)
        self.batch_size = batch_size # document number in a mini batch
        self.mini_batch_generator = Corpus.generator_mini_batch(corpus, self.batch_size) # default batch size 1000

        self.C = corpus.C  # C is number of words in the corpus, use for updating gamma for SCVB0
        self.D = corpus.D # document number in full batch
        self.K = seeds_topic_matrix.shape[1]
        self.V = corpus.V  # vocabulary size of regular words
        self.seeds_topic_matrix = seeds_topic_matrix # V x K matrix, if exists value which indicates seed word w (row w) for topic k (column k)
        self.S = torch.nonzero(self.seeds_topic_matrix.sum(axis=1)).shape[0]
        self.beta = 0.1 # hyperparameter for prior of regular topic mixture phi_r
        self.beta_sum = [self.beta * V_m for V_m in self.V]
        self.mu = 0.1 # hyperparameter for prior of seed topic mixture phi_s
        self.mu_sum = self.mu * self.S # mu_sum is a K-length vector, mu_sum[k] is the sum of mu over all seed word (S[k]
        self.pi_init = 0.7
        self.pi = torch.full([self.K], self.pi_init, dtype=torch.double, requires_grad=False, device=device) # hyperparameter weight for indicator x
        # expected tokens
        self.exp_m = torch.zeros(self.D, self.K, dtype=torch.double, requires_grad=False, device=device) # suppose a general m_dk across different modalities
        # self.exp_m = [torch.zeros(self.D, self.K, requires_grad=False, device=device) for V_m in self.V] # suppose a modality-specified m_dk
        self.exp_n = [torch.zeros(self.V[m], self.K, dtype=torch.double, requires_grad=False, device=device)
                      for m in range(modality_num)] # exp_n for differnt modality
        self.exp_s = torch.zeros(self.V[guided_modality], self.K, dtype=torch.double, requires_grad=False, device=device) # use V to represent, regular word for a topic is 0, only for guided modality
        self.exp_q_z = 0
        self.eta = 0.1
        self.initialize_tokens()
        # temporal inference component
        # self.eta = torch.rand(self.T, self.K, requires_grad=False, device=device)
        # self.alpha = self.alpha_softplus_act().to(device) # T x K
        # variational distribution for eta via amortizartion, eta is T x K matrix
        # self.eta_hidden_size = 200 # number of hidden units for rnn
        # self.eta_dropout = 0.0 # dropout rate on rnn for eta
        # self.eta_nlayers = 3 # number of layers for eta
        # self.delta = 0.01  # prior variance
        # self.q_eta_map = nn.Linear(self.V, self.eta_hidden_size)
        # self.q_eta = nn.LSTM(self.eta_hidden_size, self.eta_hidden_size, self.eta_nlayers, dropout=self.eta_dropout)
        # self.mu_q_eta = nn.Linear(self.eta_hidden_size+self.K, self.K, bias=True)
        # self.logsigma_q_eta  = nn.Linear(self.eta_hidden_size+self.K, self.K, bias=True)
        # optimizer
        # self.clip = 0
        # self.lr = 0.0001
        # self.wdecay = 1.2e-6
        # self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wdecay)
        # self.max_logsigma_t = 5.0 # avoid the value to be too big
        # self.min_logsigma_t = -5.0

    def initialize_tokens(self):
        '''
        obtain initialized tokens E[n_wk], E[s_wk], E[m_dk]
        '''
        print("Obtain initialized tokens")
        self.exp_n[0] = torch.load("./guide_prior/init_exp_n_icd.pt", map_location=device)  # get exp_n for guided ICD modality, V X K matrix
        self.exp_n[1] = torch.load("./guide_prior/init_exp_n_med.pt", map_location=device)  # get exp_n for unguided med modaltiy, V X K matrix
        self.exp_s = torch.load("./guide_prior/init_exp_s_icd.pt", map_location=device)  # get exp_s for guided ICD modality only, V X K matrix
        self.exp_m = torch.load("./guide_prior/init_exp_m.pt", map_location=device)  # get exp_m without respect to modality, D X K matrix
        self.exp_n_sum = [torch.sum(exp_n, dim=0) for exp_n in self.exp_n] # sum over w, exp_n is [V K] dimensionality, exp_n_sum is K-len vector for each modality
        self.exp_s_sum = torch.sum(self.exp_s, dim=0) # sum over w, exp_p is [V K] dimensionality, exp_s_sum is K-len vector
        self.exp_m_sum = torch.sum(self.exp_m, dim=1) # sum over k, exp_m is [D K] dimensionality, exp_m_sum is D-len vector

    def get_kl_z(self, batch_indices):
        '''
        compute the elbo excluding eta, kl with respect to eta is computed seperately after estimation by neural network
        '''
        # compute kl(q_z || p_z), should be positive
        kl_z = 0
        # E_q[log q(z | gamma)]
        kl_z = self.exp_q_z
        # E_q[ log p(z | alpha), alpha is softplus(eta)
        p_z = torch.sum(gammaln(self.eta + self.exp_m[batch_indices]), dim=1) - \
                gammaln(self.K * self.eta + self.exp_m_sum[batch_indices])
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

    def SCVB0_guided(self, batch_BOW, batch_indices, batch_C, iter_n, guided_m=0):
        # temp_exp_m = torch.zeros(self.D, self.K, device=device)
        temp_exp_m_batch = torch.zeros(batch_BOW.shape[0], self.K, device=device)
        temp_exp_n = torch.zeros(self.V[guided_m], self.K, device=device)
        temp_exp_s = torch.zeros(self.V[guided_m], self.K, device=device)
        gamma_sr_sum = torch.zeros(self.K, device=device)
        # M step
        for d_i, doc_id in enumerate(batch_indices):
            temp_gamma_ss = torch.zeros(self.V[guided_m], self.K, dtype=torch.double, device=device) #  V x K  # non-seed regular word will be zero
            temp_gamma_sr = torch.zeros(self.V[guided_m], self.K, dtype=torch.double, device=device)
            temp_gamma_rr = torch.zeros(self.V[guided_m], self.K, dtype=torch.double, device=device)
            BOW_nonzero = torch.nonzero(batch_BOW[d_i]).squeeze(dim=1)
            # seed word and seed topic
            temp_gamma_ss[BOW_nonzero] = self.seeds_topic_matrix[BOW_nonzero] * (self.exp_m[doc_id] + self.eta) \
                                         * (self.mu + self.exp_s[BOW_nonzero]) / (self.mu_sum + self.exp_s_sum) * self.pi
            # seed word but regular topic
            temp_gamma_sr[BOW_nonzero] = self.seeds_topic_matrix[BOW_nonzero] * (self.exp_m[doc_id] + self.eta) \
                                         * (self.beta + self.exp_n[guided_m][BOW_nonzero]) / (self.beta_sum[guided_m] + self.exp_n_sum[guided_m]) * (1-self.pi)
            # regular word must be regular topic
            temp_gamma_rr[BOW_nonzero] = (1-self.seeds_topic_matrix[BOW_nonzero]) * (self.exp_m[doc_id] + self.eta) \
                                         * (self.beta + self.exp_n[guided_m][BOW_nonzero]) / (self.beta_sum[guided_m] + self.exp_n_sum[guided_m])
            # normalization
            temp_gamma_s_sum = temp_gamma_ss.sum(dim=1).unsqueeze(1) + temp_gamma_sr.sum(dim=1).unsqueeze(1)
            temp_gamma_r_sum = temp_gamma_rr.sum(dim=1).unsqueeze(1)
            temp_gamma_ss /= temp_gamma_s_sum + mini_val
            temp_gamma_sr /= temp_gamma_s_sum + mini_val
            temp_gamma_rr /= temp_gamma_r_sum + mini_val
            # calculate frequency * gamma for each word in a document
            seed_word_list = torch.nonzero(self.seeds_topic_matrix[BOW_nonzero].sum(axis=1)).squeeze(1)
            non_seed_index = torch.where((1-self.seeds_topic_matrix[BOW_nonzero].sum(axis=1)) >= 0, (1-self.seeds_topic_matrix[BOW_nonzero].sum(axis=1)), 0)
            non_seed_word_list = torch.nonzero(non_seed_index).squeeze(1)
            temp_gamma_s = (self.pi*temp_gamma_ss + (1-self.pi)*(temp_gamma_sr+temp_gamma_rr))[BOW_nonzero][seed_word_list]
            temp_gamma_r = temp_gamma_rr[BOW_nonzero][non_seed_word_list]
            temp_gamma = torch.cat((temp_gamma_s, temp_gamma_r), 0)
            # calculate sufficient statistics
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
        self.exp_s = (1-rho)*self.exp_s + rho*temp_exp_s*self.C[guided_m]/batch_C
        self.exp_s_sum = torch.sum(self.exp_s, dim=0) # sum over w, exp_p is [V K] dimensionality
        self.exp_n[guided_m] = (1-rho)*self.exp_n[guided_m] + rho*temp_exp_n*self.C[guided_m]/batch_C
        self.exp_n_sum[guided_m] = torch.sum(self.exp_n[guided_m], dim=0) # sum over w, exp_n is [V K] dimensionality
        self.update_hyperparams(gamma_sr_sum) # update hyperparameters

    def SCVB0_unguided(self, batch_BOW, batch_indices, batch_C, iter_n, unguided_m):
        temp_exp_n = torch.zeros(self.V[unguided_m], self.K, device=device)
        gamma_sum = torch.zeros(self.K, device=device)
        # M step
        for d_i, doc_id in enumerate(batch_indices):
            temp_gamma = torch.zeros(self.V[unguided_m], self.K, dtype=torch.double, device=device) #  V x K
            BOW_nonzero = torch.nonzero(batch_BOW[d_i]).squeeze(dim=1)
            # regular word must be regular topic
            temp_gamma[BOW_nonzero] = (self.exp_m[doc_id] + self.eta) * (self.beta + self.exp_n[unguided_m][BOW_nonzero]) \
                                      / (self.beta_sum[unguided_m] + self.exp_n_sum[unguided_m])
            # normalization
            temp_gamma_sum = temp_gamma.sum(dim=1).unsqueeze(1)
            temp_gamma /= temp_gamma_sum + mini_val
            # calculate sufficient statistics
            temp_exp_n += temp_gamma * batch_BOW[d_i].unsqueeze(1)
        # E step
        # update expected terms
        rho = 1 / math.pow((iter_n + 5), 0.9)
        self.exp_n[unguided_m] = (1-rho)*self.exp_n[unguided_m] + rho*temp_exp_n*self.C[unguided_m]/batch_C
        self.exp_n_sum[unguided_m] = torch.sum(self.exp_n[unguided_m], dim=0) # sum over w, exp_n is [V K] dimensionality

    def update_hyperparams(self, gamma_sr_sum):
        '''
        update hyperparameters pi using Bernoulli trial
        '''
        self.pi = self.exp_s_sum / (self.exp_s_sum + gamma_sr_sum + mini_val)
        # fill pi_init for pi_k with non-computed topic k (as gamma is 0) or very low value
        self.pi = torch.where(self.pi > 0.1, self.pi, torch.ones(self.K, dtype=torch.double, device=device)*self.pi_init)

    def inference_SCVB_SGD(self, max_epoch=10, save_every=1):
        '''
        inference algorithm for dynamic seed-guided topic model, apply stochastic collaposed variational inference for latent variable z,
        and apply stochastic gradient descent for dynamic variables \eta (\alpha)
        '''
        elbo = [0,]
        for epoch in range(0, max_epoch):
            print("Training for epoch", epoch)
            batch_n = (self.D // self.batch_size) + 1
            for i, d in enumerate(self.mini_batch_generator): # For each epoach, we sample a series of mini_batch data once
                print("Running for %d minibatch", i)
                # start_time = time.time()
                batch_docs, batch_indices, batch_C = d  # batch_C is total number of words within a minibatch for SCVB0
                # batch_BOW = torch.zeros(len(batch_docs), self.V, dtype=torch.int, requires_grad=False, device=device) # document number (not M) x V
                for m in range(self.modaltiy_num):
                    # modaltiy specific BOW matrix, shape is D X V[m]
                    batch_BOW_m = torch.zeros(len(batch_docs), self.V[m], dtype=torch.int, requires_grad=False, device=device)  # document number (not M) x V
                    batch_C_m = sum([doc_C[m] for doc_C in batch_C])
                    for d_i, (doc_id, doc) in enumerate(zip(batch_indices, batch_docs)):
                        for word_id, freq in doc.words_dict[m].items():
                            batch_BOW_m[d_i, word_id] = freq
                    if m == self.guided_modality:
                        self.SCVB0_guided(batch_BOW_m, batch_indices, batch_C_m, epoch*batch_n+i, guided_m=0)
                    else:
                        self.SCVB0_unguided(batch_BOW_m, batch_indices, batch_C_m, epoch*batch_n+i, unguided_m=m)
                    # kl_z = self.get_kl_z(batch_indices)
                    # loss = kl_z
                    # loss *= self.C / batch_C
                    # print("took %s seconds for minibatch %s" % (time.time() - start_time, epoch))
                    # elbo.append(loss.detach().cpu().numpy().item())
                    # print("epoch: %s, minibatch %s, elbo: %s, elbo diff: %s" % (epoch, i, elbo[-1], np.abs(elbo[-1] - elbo[-2])))
                    self.exp_q_z = 0 # update to zero for next epoch
                self.save_parameters(epoch)
        return elbo


    def save_parameters(self, epoch):
        torch.save(self.exp_m, "./parameters/exp_m_%s.pt" % (epoch))
        torch.save(self.exp_n[0], "./parameters/exp_n_icd_%s.pt" % (epoch))
        torch.save(self.exp_n[1], "./parameters/exp_n_med_%s.pt" % (epoch))
        torch.save(self.exp_s, "./parameters/exp_s_%s.pt" % (epoch))
        torch.save(self.pi, "./parameters/pi_%s.pt" % (epoch))

