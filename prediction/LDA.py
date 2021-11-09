import os, time
import csv
import numpy as np
import math 
# import seaborn as sns
import pickle
import scipy
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, average_precision_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit 
from IPython import embed

def classicLDA(X, label, diseases, f=None, n_topics=75):
    start_time = time.time()
    print('classic LDA (with {} topics) + EN: '.format(n_topics))
    print((X.shape, label.shape))

    model = LatentDirichletAllocation(n_components=n_topics, learning_method='online')
    m_time = time.time()
    X_mixture = model.fit_transform(X)
    print('LDA perplexity: {}  time: {:.2f} s'.format(model.perplexity(X), time.time()-m_time))
    
    # parameters = {'penalty': ['l1', 'l2', 'elasticnet', 'none']}
    parameters = {'C':[10, 1, 0.1], 'l1_ratio': [0, 0.2, 0.5, 0.8, 1]}
    clf = []
    AUPRCs, AUROCs = [], []
    for i, dis in enumerate(diseases):
        print((i, dis))
        s_time = time.time()
        y = label[:,i]
        spliter = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
        clf_cv = GridSearchCV(LogisticRegression(penalty='elasticnet', solver='saga'), parameters, scoring='average_precision', cv=spliter.split(X, y), n_jobs=-1)
        clf_cv.fit(X, y)
        print(clf_cv.best_params_)
        print('cross-validation time: {:.2f} s  best_score: {:.1%}'.format(time.time()-s_time, clf_cv.best_score_))
        if f:
            print(clf_cv.best_params_, file=f)
            print('cross-validation time: {:.2f} s  best_score: {:.1%}'.format(time.time()-s_time, clf_cv.best_score_), file=f)

        clf.append(clf_cv.best_estimator_)
        p_time = time.time()
        clf[i].fit(X_mixture, y)
        prob = clf[i].predict(X_mixture)
        auprc = average_precision_score(label[:, i], prob)
        auroc = roc_auc_score(label[:, i], prob)
        print(' AUPRC of {}: {:.1%}, AUROC: {:.1%}  time: {:.2f} s'.format(dis, auprc, auroc, time.time()-p_time))
        print(' AUPRC of {}: {:.1%}, AUROC: {:.1%}  time: {:.2f} s'.format(dis, auprc, auroc, time.time()-p_time), file=f)
        AUPRCs.append(auprc)
        AUROCs.append(auroc)

    print('training time: {:.2f} s'.format(time.time()-start_time))
    print('training time: {:.2f} s'.format(time.time()-start_time), file=f)
    return (model, clf), [AUPRCs, AUROCs]

def test_classicLDA(model_name, model, clf, test_X, test_label, diseases, f=None):
    test_X_mixture = model.transform(test_X)
    print('test of {} + EN'.format(model_name))
    AUPRCs, AUROCs = [], []
    for i, dis in enumerate(diseases):
        prob = clf[i].predict(test_X_mixture)
        auprc = average_precision_score(test_label[:, i], prob)
        auroc = roc_auc_score(test_label[:, i], prob)
        print(' AUPRC of {}: {:.1%},  AUROC: {:.1%}'.format(dis, auprc, auroc))
        if f:
            print(' AUPRC of {}: {:.1%},  AUROC: {:.1%}'.format(dis, auprc, auroc), file=f)
        AUPRCs.append(auprc)
        AUROCs.append(auroc)
    return [AUPRCs, AUROCs]
        



