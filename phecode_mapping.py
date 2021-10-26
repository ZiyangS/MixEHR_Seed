import json
import pandas as pd
import time


phecode_icd9_map = pd.read_csv('phecode_mapping/phecode_icd9_rolled.csv')
# print(len(phecode_icd9_map['Excl. Phecodes'].unique())) # 233
phecode_icd9_map = phecode_icd9_map[phecode_icd9_map.Leaf.notna()] # delele Leaf node is not 0 or 1, abnormal row
# phecode_icd9_map.phecode = phecode_icd9_map.ICD9.astype(str)
# phecode_icd9_map.phecode = phecode_icd9_map.PheCode.astype(str)
# read parent phecode with Leaf = 0, and child phecode with Leaf = 1
parent_phecodes = phecode_icd9_map.loc[phecode_icd9_map['Leaf'] == 0]
parent_phecodes = list(parent_phecodes.PheCode.unique()) # 505
child_phecodes = phecode_icd9_map.loc[phecode_icd9_map['Leaf'] == 1]
child_phecodes = list(child_phecodes.PheCode.unique()) # 1360
unique_icd9 = list(phecode_icd9_map.ICD9.unique()) # 15558

# In rolled phecode-icd mapping, each icd only maps to a single phecode, thus we obtain 15557 icd codes for 1865 phecodes
parent_phecode_ICD_mapping = {}
child_phecode_ICD_mapping = {}
for icd in unique_icd9:
    rows = phecode_icd9_map.loc[phecode_icd9_map['ICD9'] == icd]
    map_phecodes = list(rows.PheCode)[0]
    if map_phecodes in parent_phecodes:
        if map_phecodes not in parent_phecode_ICD_mapping.keys():
            parent_phecode_ICD_mapping[map_phecodes] = [icd]
        else:
            parent_phecode_ICD_mapping[map_phecodes].append(icd)
    elif map_phecodes in child_phecodes:
        if map_phecodes not in child_phecode_ICD_mapping.keys():
            child_phecode_ICD_mapping[map_phecodes] = [icd]
        else:
            child_phecode_ICD_mapping[map_phecodes].append(icd)

parent_ICDs = set()
for keys, values in parent_phecode_ICD_mapping.items():
    for icd in values:
        print("parent", keys, icd)
        parent_ICDs.add(icd)
child_ICDs = set()
for keys, values in child_phecode_ICD_mapping.items():
    for icd in values:
        print("child", keys, icd)
        child_ICDs.add(icd)
print(len(parent_ICDs))
print(len(child_ICDs))
print(len(child_ICDs & parent_ICDs))
print(parent_ICDs)
print(child_ICDs)

phecode_icd_items = list(zip(phecode_icd9_map.PheCode, phecode_icd9_map.ICD9)) # get all (phecode, icd9) pairs
print(len(phecode_icd_items)) # 15557, same as the rows of phecode_icd9_map
phecode_icd_dict = {} # each phecode is a key, the value corresponds to a key is [ICD9, ... , ICD9]
for pc, icd in phecode_icd_items:
    if pc not in phecode_icd_dict.keys():
        phecode_icd_dict[pc] = [icd]
    else:
        phecode_icd_dict[pc].append(icd)
print(len(phecode_icd_dict.keys())) # 1865
with open('phecode_mapping\/full_phecode_icd_dict.json', 'w') as fp:
    json.dump(phecode_icd_dict, fp)
