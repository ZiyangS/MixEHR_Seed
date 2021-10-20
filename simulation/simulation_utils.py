import json

def keystoint(x):
    '''
    convert key to int datatype
    '''
    return {float(k): v for k, v in x.items()}

def simulation_tokenize_phecode_icd():
    '''
    tokenization, map phecode and icd code from 0 to K-1/V-1
    :return: mapped_phecode, mapped_icd, tokenized_phecode_icd
    '''
    with open("simulation/select_tokenized_phecode_icd.json", "r") as read_file:
        phecode_icd_dict = json.load(read_file, object_hook=keystoint)
    # print(phecode_icd_dict)
    # print(len(phecode_icd_dict)) # 648
    mapped_phecode = {} # key is phecode, value is the mapped index of phecode from 1 to K-1
    mapped_icd = {} # key is icd, value is the mapped index of icd from 1 to V-1
    unique_icd = set()
    for i, key in enumerate(phecode_icd_dict):
        mapped_phecode[key] = i
        for value in phecode_icd_dict[key]:
            unique_icd.add(value)
    # print(mapped_phecode)
    # print(len(unique_icd)) # unique_word is 11746
    for i, value in enumerate(unique_icd):
        mapped_icd[value] = i
    # print(mapped_icd)
    tokenized_phecode_icd = {mapped_phecode[key]: [mapped_icd[v] for v in value] for key, value in phecode_icd_dict.items()}
    # print(tokenized_phecode_icd)
    return mapped_phecode, mapped_icd, tokenized_phecode_icd

if __name__ == "__main__":
    mapped_phecode, mapped_icd, tokenized_phecode_icd = simulation_tokenize_phecode_icd()
    print(tokenized_phecode_icd)