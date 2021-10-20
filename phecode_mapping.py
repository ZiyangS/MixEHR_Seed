import json
import numpy as np
import pandas as pd

phecode_definitions = pd.read_csv('phecode_mapping/phecode_definitions1.2.csv')
phecode_definitions.phecode = phecode_definitions.phecode.astype(str)
parent_phecodes = phecode_definitions.loc[phecode_definitions['leaf'] == 0] # 506
parent_phecodes = list(parent_phecodes.phecode)
child_phecodes = phecode_definitions.loc[phecode_definitions['leaf'] == 1] # 1360
child_phecodes = list(child_phecodes.phecode)

phecode_icd9_map = pd.read_csv('phecode_mapping/phecode_icd9_map_unrolled.csv')
# print(phecode_icd9_map.icd9.dtypes) # object, mix
# print(phecode_icd9_map.phecode.dtypes) # float
phecode_icd9_map.phecode = phecode_icd9_map.phecode.astype(str)
# print(len(phecode_icd9_map.icd9.unique())) # 13707
# print(len(phecode_icd9_map.phecode.unique())) # 1817, less than 506 + 1360, some phecodes do not have correspondence
unique_icd9 = list(phecode_icd9_map.icd9.unique())
unique_phecode = list(phecode_icd9_map.phecode.unique())

# phecodes are grouped as three level (child, two-level or one-level parent)
# way 1: parent phecode as a topic, we only consider highest-level parent phecode as a topic
# way 2: child phecode as a topic

parent_phecode_set = set() # highest-level parent phecode, 793
nonuse_parent_phecode_set = set() # lower-level parent phecode, 131
child_phecode_set= set() # child phecode, 1685
for icd in unique_icd9:
    rows = phecode_icd9_map.loc[phecode_icd9_map['icd9'] == icd]
    rows_phecodes = list(rows.phecode)
    if len(rows_phecodes) == 1: # only have one icd-phecode mapping, can only use this one
        parent_phecode_set.add(rows_phecodes[0])
        child_phecode_set.add(rows_phecodes[0])
    if len(rows_phecodes) == 2:
        for pc in rows_phecodes:
            flag = False
            if pc in parent_phecodes:
                parent_phecode_set.add(pc)
            else:
                child_phecode_set.add(pc)
    if len(rows_phecodes) == 3:
        for pc in rows_phecodes:
            if pc in parent_phecodes and pc.split('.')[1] == '0':
                parent_phecode_set.add(pc)
            elif pc in parent_phecodes:
                nonuse_parent_phecode_set.add(pc)
            else:
                child_phecode_set.add(pc)

print(len(parent_phecode_set))
print(len(nonuse_parent_phecode_set))
print(len(parent_phecode_set.union(nonuse_parent_phecode_set)))
print(len(parent_phecode_set - nonuse_parent_phecode_set))
new_parent_phecode_set = parent_phecode_set - nonuse_parent_phecode_set # subtract non-highest parent phecode

for icd in unique_icd9:
    rows = phecode_icd9_map.loc[phecode_icd9_map['icd9'] == icd]
    rows_phecodes = list(rows.phecode)
    # flag = False
    for pc in rows_phecodes: # check whether all icd codes are distributed in parent_phecode_set/child_phecode_set
        # if pc in parent_phecode_set:
        #     flag = True
        # if pc in child_phecode_set:
        #     flag = True
        if pc in new_parent_phecode_set:
            flag = True
    # print(flag)
    # if flag == False:
    #     print(icd, rows_phecodes)


phecode_icd_items = list(zip(phecode_icd9_map.phecode, phecode_icd9_map.icd9)) # get all (phecode, icd9) pairs
# 20783, same as the rows of phecode_icd9_map
parent_phecode_icd_dict = {} # each parent phecode is a key, the value corresponds to a key is [ICD9, ... , ICD9]
child_phecode_icd_dict = {} # each child phecode is a key, the value corresponds to a key is [ICD9, ... , ICD9]
new_parent_phecode_dict = {}
count = 0
for pc, icd in phecode_icd_items:
    if pc in parent_phecode_set:
        if pc not in parent_phecode_icd_dict.keys():
            parent_phecode_icd_dict[pc] = [icd]
        else:
            parent_phecode_icd_dict[pc].append(icd)
    if pc in child_phecode_set:
        if pc not in child_phecode_icd_dict.keys():
            child_phecode_icd_dict[pc] = [icd]
        else:
            child_phecode_icd_dict[pc].append(icd)
    if pc in new_parent_phecode_set:
        if pc not in new_parent_phecode_dict.keys():
            new_parent_phecode_dict[pc] = [icd]
        else:
            new_parent_phecode_dict[pc].append(icd)

with open('phecode_mapping\parent_phecode_icd_dict.json', 'w') as fp:
    json.dump(parent_phecode_icd_dict, fp)
with open('phecode_mapping\child_phecode_icd_dict.json', 'w') as fp:
    json.dump(child_phecode_icd_dict, fp)
with open('phecode_mapping\/new_parent_phecode_dict.json', 'w') as fp:
    json.dump(new_parent_phecode_dict, fp)
print(len(parent_phecode_icd_dict.keys()))
print(len(new_parent_phecode_dict.keys()))