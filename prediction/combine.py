import numpy as np
import pickle as pkl
import pandas as pd 
from IPython import embed

models = ['SVM', 'LASSO', 'ridge', 'RF_tmp']  # 'RF'
data_auprc = []
data_auroc = []
for m in models:
    print(m)
    result_i = np.load(open(m+'_scores.npy', 'rb'), allow_pickle=True).item()
    score_i = np.array(result_i['test']).mean(0)
    print(score_i)
    data_auprc.append(score_i[0])
    data_auroc.append(score_i[1])

data_auprc = np.array(data_auprc)
data_auroc = np.array(data_auroc)
embed()

column = ['adhd', 'ami', 'anxiodepressive', 'asthma', 'autism', 'chf', 'copd', 'diabetes', 'epilepsy', 'hiv', 'hypertension', 'ihd', 'schizophrenia', 'stroke']
index = ['ratio of positive samples', 'SVM', 'LASSO', 'ridge', 'RF']
ratio = [0.02168054,0.02867335,0.32473904,0.09585276,0.003913,  0.03480691, 0.06558661, 0.08279001,0.00963188,0.0015859, 0.19279932,0.08716521, 0.01346976,0.03059421]

result_auprc = pd.DataFrame(data=np.vstack([ratio, data_auprc]), index=index, columns=column)
result_auroc = pd.DataFrame(data=np.vstack([ratio, data_auroc]), index=index, columns=column)

pkl.dump(result_auprc, open('baselines_auprc.pkl', 'wb'))
pkl.dump(result_auroc, open('baselines_auroc.pkl', 'wb'))