import re
import sys
import numpy as np
from IPython import embed

logfile = sys.argv[1]
with open(logfile, 'r') as f:
    text = f.readlines()


labels = ['hiv',	'autism',	'epilepsy',	'schizophrenia',	'adhd',	'ami',	'stroke',	'chf',	'copd',	'diabetes',	'ihd',	'asthma',	'hypertension',	'anxiodepressive']

auprc = {'train': {}, 'test': {}}
auroc = {'train': {}, 'test': {}}
for l in labels:
    auprc['train'][l], auprc['test'][l] = [], []
    auroc['train'][l], auroc['test'][l] = [], []

it = -1
phase = 'train'
for line in text[0:]:
    if line.split()[0] == 'iter':
        it += 1
        phase = 'train'
    elif line.split()[0] == 'training':
        phase = 'test'

    if 'AUPRC' in line and 'AUROC' in line:
        l = line.split()[2].rstrip(':')
        nums = re.findall("(?<=[AZaz])?(?!\d*=)[0-9.+-]+",line)
        auprc[phase][l].append(float(nums[0]))
        auroc[phase][l].append(float(nums[1]))

for l in labels:
    for phase in ['train', 'test']:
        auprc[phase][l] = np.array(auprc[phase][l])
        auroc[phase][l] = np.array(auroc[phase][l])


print(logfile + ' test: ')
for l in labels:
    print(' {} auprc: {:.1f} +- {:.1f}  auroc: {:.1f} +- {:.1f}'.format(l, auprc['test'][l].mean(), auprc['test'][l].std(), auroc['test'][l].mean(), auroc['test'][l].std()))
