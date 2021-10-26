import json

def keystoint(x):
    '''
    convert key to int datatype
    '''
    return {float(k): v for k, v in x.items()}

def tokenize_phecode_icd():
    '''
    tokenization, map phecode and icd code from 0 to K-1/V-1
    :return: mapped_phecode, mapped_icd, tokenized_phecode_icd
    '''
    with open("./phecode_mapping/full_phecode_icd_dict.json", "r") as read_file:
        phecode_icd_dict = json.load(read_file, object_hook=keystoint)
    # print(len(phecode_icd_dict)) # 1865
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

def tokenize_phecode_icd_corpus(icd_in_corpus):
    '''
    tokenization, map phecode and icd code from 0 to K-1/V-1 for a given corpus
    :return: mapped_phecode, mapped_icd, tokenized_phecode_icd
    '''
    with open("./phecode_mapping/full_phecode_icd_dict.json", "r") as read_file:
        phecode_icd_dict = json.load(read_file, object_hook=keystoint)
    # print(len(phecode_icd_dict)) # number of phecodes is 1865
    all_icds_US = []
    for icds in phecode_icd_dict.values():
        all_icds_US.extend(icds)
    # print(len(all_icds_US)) # number of ICD codes (US) is 15557

    # get ICD codes and phecodes appear in the corpus (CA)
    icds_US_in_corpus = []
    icds_US2CA = {}
    icds_regular = []
    for icd in icd_in_corpus:
        if len(icd) == 3:  # all 3-digit icd codes in corpus are shown in phecode-icd mapping file
            icds_US_in_corpus.append(icd)
            icds_US2CA[icd] = icd
        else:
            icd_US = icd[0:3] + '.' + icd[3]
            if icd_US in all_icds_US:
                icds_US_in_corpus.append(icd_US)
                icds_US2CA[icd_US] = icd
            else:
                icds_regular.append(icd)
    # print(len(icds_US_in_corpus))  # we only find 5741 ICD codes for phecode-icd (US) mapping files
    # print(len(icd_in_corpus))  # the documents contain 8539 ICD codes (CA), other codes are assumed as regular word
    phecode_icd_dict_corpus = {} # get a new phecode-icd mapping with the ICD codes (CA) in corpus and all associated phecodes
    for i, (key, values) in enumerate(phecode_icd_dict.items()):
        for icd in values:
            if icd in icds_US_in_corpus:
                if key not in phecode_icd_dict_corpus.keys():
                    phecode_icd_dict_corpus[key] = [icd]
                else:
                    phecode_icd_dict_corpus[key].append(icd)
    # print(len(phecode_icd_dict_corpus.keys())) # number of phecodes are 1569 and number of ICD (US) are 5741

    # tokenization for icd codes (CA) and phecodes which appear in the codes
    mapped_phecode = {} # key is phecode, value is the mapped index of phecode from 1 to K-1, K is 1569
    mapped_icd = {} # key is icd, value is the mapped index of icd from 1 to V-1, V is 8539 as seed and regular words are included
    for i, key in enumerate(phecode_icd_dict_corpus):
        mapped_phecode[key] = i
    for i, value in enumerate(icd_in_corpus): # len(icd_in_corpus) is 8539
        mapped_icd[value] = i
    tokenized_phecode_icd = {mapped_phecode[key]: [mapped_icd[icds_US2CA[v]] for v in value] for key, value in phecode_icd_dict_corpus.items()}
    # len(key) is 1569, len(values) is 5741
    return mapped_phecode, mapped_icd, tokenized_phecode_icd


if __name__ == "__main__":
    mapped_phecode, mapped_icd, tokenized_phecode_icd = tokenize_phecode_icd()
    print(tokenized_phecode_icd)