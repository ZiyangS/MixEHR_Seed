import os, time
import csv
import numpy as np
import math 
# import seaborn as sns
import pickle
import scipy
from sklearn.svm import *
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, average_precision_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit 
from IPython import embed

def SVM(X, label, diseases, f=None):
    start_time = time.time()
    print('SVM: ')
    parameters = {'C':[0.1, 0.2, 0.4, 0.6, 0.8]}
    clf = []
    AUPRCs, AUROCs = [], []
    
    for i, dis in enumerate(diseases):
        print((i, dis))
        s_time = time.time()
        y = label[:,i]
        spliter = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
        clf_cv = GridSearchCV(LinearSVC(class_weight='balanced'), parameters, scoring='average_precision',
                              cv=spliter.split(X, y), n_jobs=-1)
        clf_cv.fit(X, y)
        print(clf_cv.best_params_)
        print('cross-validation time: {:.2f} s  best_score: {:.1%}'.format(time.time()-s_time, clf_cv.best_score_))
        if f:
            print(clf_cv.best_params_, file=f)
            print('cross-validation time: {:.2f} s  best_score: {:.1%}'.format(time.time()-s_time, clf_cv.best_score_), file=f)

        clf.append(clf_cv.best_estimator_)
        p_time = time.time()
        clf[i].fit(X, y)
        prob = clf[i].decision_function(X)
        auprc = average_precision_score(label[:, i], prob)
        auroc = roc_auc_score(label[:, i], prob)
        print(' AUPRC of {}: {:.1%}, AUROC: {:.1%}  time: {:.2f} s'.format(dis, auprc, auroc, time.time()-p_time))
        if f:
            print(' AUPRC of {}: {:.1%}, AUROC: {:.1%}  time: {:.2f} s'.format(dis, auprc, auroc, time.time()-p_time), file=f)
        AUPRCs.append(auprc)
        AUROCs.append(auroc)

    print('training time: {:.2f} s'.format(time.time()-start_time))
    if f:
        print('training time: {:.2f} s'.format(time.time()-start_time), file=f)
    return clf, [AUPRCs, AUROCs]

def test_SVM(model_name, model, test_X, test_label, diseases, f=None):
    print('test of {}'.format(model_name))
    AUPRCs, AUROCs = [], []
    for i, dis in enumerate(diseases):
        prob = model[i].decision_function(test_X)
        auprc = average_precision_score(test_label[:, i], prob)
        auroc = roc_auc_score(test_label[:, i], prob)
        print(' AUPRC of {}: {:.1%},  AUROC: {:.1%}'.format(dis, auprc, auroc))
        if f:
            print(' AUPRC of {}: {:.1%},  AUROC: {:.1%}'.format(dis, auprc, auroc), file=f)
        AUPRCs.append(auprc)
        AUROCs.append(auroc)
    return [AUPRCs, AUROCs]
        



