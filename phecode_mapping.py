import pandas as pd
import numpy as np

phecode_icd9_map = pd.read_csv('phecode_mapping/phecode_icd9_map_unrolled.csv')
phecode_icd9_map.phecode = phecode_icd9_map.phecode.astype(str)
print(phecode_icd9_map)
print(len(phecode_icd9_map.icd9.unique())) # 13707
print(len(phecode_icd9_map.phecode.unique())) # 1817

unique_icd9 = list(phecode_icd9_map.icd9.unique())
unique_phecode = list(phecode_icd9_map.phecode.unique())

# two version
# first version, 3 digit phecode as a topic
# second version, 4 digit phecode as a topic

# first version, 544
unique_phecode_3digit = [phecode.split('.')[0] for phecode in unique_phecode]
unique_phecode_3digit = list(set(unique_phecode_3digit))
print(unique_phecode)
print(unique_phecode_3digit)
print(len(unique_phecode_3digit))

phecode_icd_items = list(zip(phecode_icd9_map.phecode, phecode_icd9_map.icd9))
# print(phecode_icd_items)
# print(len(phecode_icd_items)) # 20783
phecode_icd_dict = {}
count = 0
for key, value in phecode_icd_items:
    if key.split('.')[1] != '0':
        continue
    count +=1
    if key not in phecode_icd_dict.keys():
        phecode_icd_dict[key] = [value]
    else:
        phecode_icd_dict[key].append(value)
# print(phecode_icd_dict)
print(len(phecode_icd_dict.keys()))
print(count)
print(len(phecode_icd_dict['8.0']))
print(phecode_icd_dict.keys())# still have one ,maybe  305.2 305.21 there is no 305.0

print(phecode_icd_dict)
import json
with open('phecode_mapping\phecode3_icd_dict.json', 'w') as fp:
    json.dump(phecode_icd_dict, fp)

# print((phecode_icd_dict['8.0']))
#
# print(len(phecode_icd_dict['8.5']))
# print((phecode_icd_dict['8.5']))
# print(phecode_icd_dict.keys())

# second version, 1817

# print(111111111111111111111)
# phecode_icd_items = list(zip(phecode_icd9_map.phecode, phecode_icd9_map.icd9))
# phecode_icd_dict_4digit = {}
# count = 0
# for key, value in phecode_icd_items:
#     count += 1
#     if key not in phecode_icd_dict_4digit.keys():
#         phecode_icd_dict_4digit[key] = [value]
#     else:
#         phecode_icd_dict_4digit[key].append(value)
#
# # create 3 digit - 4 digit map
# for key, value in phecode_icd_items:
#     if key.split('.')[1] != '0':
#         phecode_3digit = key.split('.')[0] + '.0'
#         print(key,value)
#         print(phecode_3digit)
#         print(phecode_icd_dict_4digit[phecode_3digit])
#         if value in phecode_icd_dict_4digit[phecode_3digit]:
#             phecode_icd_dict_4digit[phecode_3digit].remove(value)
#             count -= 1
#
# print(len(phecode_icd_dict_4digit.keys()))
# print(count)
# print(len(phecode_icd_dict_4digit['8.0']))