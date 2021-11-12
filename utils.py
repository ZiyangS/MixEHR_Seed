import json
import torch

def keystoint(x):
    '''
    convert key to int datatype
    '''
    return {float(k): v for k, v in x.items()}

def tokenize_parent_phecode_icd_corpus(icd_in_corpus):
    '''
    tokenization, map parent phecodes and icd codes from 0 to K-1/V-1 for a given corpus
    :return: mapped_phecode, mapped_icd, tokenized_phecode_icd
    '''
    with open("./phecode_mapping/parent_phecode_icd_dict_rolled.json", "r") as read_file:
        phecode_icd_dict = json.load(read_file, object_hook=keystoint)
    # print(len(phecode_icd_dict)) # number of phecodes is 586
    all_icds_US = []
    for icds in phecode_icd_dict.values():
        all_icds_US.extend(icds)
    # print(len(all_icds_US)) # number of ICD codes (US) is 15558

    # get ICD codes and phecodes appear in the corpus (CA)
    icds_US_in_corpus = []
    icds_CA_found = []
    icds_US2CA = {}
    icds_regular = []
    for icd_CA in icd_in_corpus:
        # all 3-digit CA ICD codes in corpus can be found as US ICD code
        if len(icd_CA) == 3:
            icd_US = icd_CA
            icds_US_in_corpus.append(icd_US)
            icds_CA_found.append(icd_CA)
            if icd_US in icds_US2CA.keys():
                icds_US2CA[icd_US].extend([icd_CA])
            else:
                icds_US2CA[icd_US] = [icd_CA]
        # 4-digit CA ICD codes in corpus with a end of 0
        elif len(icd_CA) == 4 and icd_CA[3] == '0': # can be found as US ICD code
            if icd_CA[0:3] + '.' + icd_CA[3] in all_icds_US: # can be found as US ICD code
                # for example, 1140 CA ICD code is Primary coccidioidomycosis (pulmonary), 114.0 US ICD code is Primary coccidioidomycosis (pulmonary)
                icd_US = icd_CA[0:3] + '.' + icd_CA[3]
                icds_US_in_corpus.append(icd_US)
                icds_CA_found.append(icd_CA)
                if icd_US in icds_US2CA.keys():
                    icds_US2CA[icd_US].extend([icd_CA])
                else:
                    icds_US2CA[icd_US] = [icd_CA]
            elif icd_CA[0:3] in all_icds_US: # can not be found as US ICD code directly
                # for example, 4630 CA ICD code is TONSILLITIS, ACUTE, 463 US ICD code is Acute tonsillitis
                icd_US = icd_CA[0:3]
                icds_US_in_corpus.append(icd_US)
                icds_CA_found.append(icd_CA)
                if icd_US in icds_US2CA.keys():
                    icds_US2CA[icd_US].extend([icd_CA])
                else:
                    icds_US2CA[icd_US] = [icd_CA]
            else:
                icds_regular.append(icd_CA)
        else:
            if icd_CA[0:3] + '.' + icd_CA[3] in all_icds_US: # can be found as US ICD code, this is the basic matching condition
                icd_US = icd_CA[0:3] + '.' + icd_CA[3]
                icds_US_in_corpus.append(icd_US)
                icds_CA_found.append(icd_CA)
                if icd_US in icds_US2CA.keys():
                    icds_US2CA[icd_US].extend([icd_CA])
                else:
                    icds_US2CA[icd_US] = [icd_CA]
            elif icd_CA[0:3] in all_icds_US:  # can not be found as US ICD code directly
                # for example, 4629 CA ICD code is Acute pharyngitis, acute pharyngitis, 462 US ICD code is Acute pharyngitis
                # but most of CA ICD codes in this category do not have corresponding meanings
                icd_US = icd_CA[0:3]
                icds_US_in_corpus.append(icd_US)
                icds_CA_found.append(icd_CA)
                if icd_US in icds_US2CA.keys():
                    icds_US2CA[icd_US].extend([icd_CA])
                else:
                    icds_US2CA[icd_US] = [icd_CA]
            else:
                icds_regular.append(icd_CA)
    # print(len(icds_US_in_corpus))  # we find 7718 CA ICD codes in phecode-ICD mapping files
    # print(len(icds_CA_found)) # we find 7718 CA ICD codes in phecode-ICD mapping files
    # print(len(icd_in_corpus))  # the documents contain 8539 CA ICD codes, other ~800~ ICD codes are assumed as regular word

    phecode_icd_dict_corpus = {} # get a new phecode-icd mapping,# key is phecode, value is CA ICD codes
    for i, (key, values) in enumerate(phecode_icd_dict.items()): # key is phecode, value is US ICD codes
        for icd_US in values:
            if icd_US in icds_US_in_corpus:
                icds_CA = icds_US2CA[icd_US]
                if key not in phecode_icd_dict_corpus.keys():
                    phecode_icd_dict_corpus[key] = icds_CA
                else:
                    phecode_icd_dict_corpus[key].extend(icds_CA)
    # print(len(phecode_icd_dict_corpus.keys())) # number of phecodes are 570, number of CA ICD codes are 7718

    # tokenization for CA ICD codes and phecodes which appear in CORPUS
    mapped_phecode = {} # key is phecode, value is the mapped index of phecode from 1 to K-1, K is 1569
    for i, key in enumerate(phecode_icd_dict_corpus): # 570, key is phecode, value is mapped index
        mapped_phecode[key] = i
    mapped_icd = {} # key is icd, value is the mapped index of icd from 1 to V-1, V is 8539 as seed and regular words are included
    for i, value in enumerate(icd_in_corpus): # 8539, key is CA ICD code, value is mapped index
        mapped_icd[value] = i
    tokenized_phecode_icd = {mapped_phecode[key]: [mapped_icd[ICD_CA] for ICD_CA in value] for key, value in phecode_icd_dict_corpus.items()}
    # len(key) is 570, len(values) is 7718

    # save phecode ICD mapping in corpus as a torch matrix
    K = len(tokenized_phecode_icd.keys())
    icd_list = mapped_icd.keys()
    V = len(icd_list)
    # print(K, V) # K is 570, V is 8539
    seeds_topic_matrix = torch.zeros(V, K, dtype=torch.int) # 8539 x 570
    for k, w_l in tokenized_phecode_icd.items():
        for w in w_l:
            seeds_topic_matrix[w, k] = 1
    # print(seeds_topic_matrix.sum()) # 7718 as 7718 words are seed words across topics
    torch.save(seeds_topic_matrix, "./phecode_mapping/parent_seed_topic_matrix.pt")
    return mapped_phecode, mapped_icd, tokenized_phecode_icd


def tokenize_all_phecode_icd_corpus(icd_in_corpus):
    '''
    tokenization, map all (parent and child) phecodes and icd codes from 0 to K-1/V-1 for a given corpus
    :return: mapped_phecode, mapped_icd, tokenized_phecode_icd
    '''
    with open("./phecode_mapping/all_phecode_icd_dict_rolled.json", "r") as read_file:
        phecode_icd_dict = json.load(read_file, object_hook=keystoint)
    # print(len(phecode_icd_dict)) # number of phecodes is 1866
    all_icds_US = []
    for icds in phecode_icd_dict.values():
        all_icds_US.extend(icds)
    # print(len(all_icds_US)) # number of ICD codes (US) is 15558

    # get ICD codes and phecodes appear in the corpus (CA)
    icds_US_in_corpus = []
    icds_CA_found = []
    icds_US2CA = {}
    icds_regular = []
    for icd_CA in icd_in_corpus:
        # all 3-digit CA ICD codes in corpus can be found as US ICD code
        if len(icd_CA) == 3:
            icd_US = icd_CA
            icds_US_in_corpus.append(icd_US)
            icds_CA_found.append(icd_CA)
            if icd_US in icds_US2CA.keys():
                icds_US2CA[icd_US].extend([icd_CA])
            else:
                icds_US2CA[icd_US] = [icd_CA]
        # 4-digit CA ICD codes in corpus with a end of 0
        elif len(icd_CA) == 4 and icd_CA[3] == '0': # can be found as US ICD code
            if icd_CA[0:3] + '.' + icd_CA[3] in all_icds_US: # can be found as US ICD code
                # for example, 1140 CA ICD code is Primary coccidioidomycosis (pulmonary), 114.0 US ICD code is Primary coccidioidomycosis (pulmonary)
                icd_US = icd_CA[0:3] + '.' + icd_CA[3]
                icds_US_in_corpus.append(icd_US)
                icds_CA_found.append(icd_CA)
                if icd_US in icds_US2CA.keys():
                    icds_US2CA[icd_US].extend([icd_CA])
                else:
                    icds_US2CA[icd_US] = [icd_CA]
            elif icd_CA[0:3] in all_icds_US: # can not be found as US ICD code directly
                # for example, 4630 CA ICD code is TONSILLITIS, ACUTE, 463 US ICD code is Acute tonsillitis
                icd_US = icd_CA[0:3]
                icds_US_in_corpus.append(icd_US)
                icds_CA_found.append(icd_CA)
                if icd_US in icds_US2CA.keys():
                    icds_US2CA[icd_US].extend([icd_CA])
                else:
                    icds_US2CA[icd_US] = [icd_CA]
            else:
                icds_regular.append(icd_CA)
        else:
            if icd_CA[0:3] + '.' + icd_CA[3] in all_icds_US: # can be found as US ICD code, this is the basic matching condition
                icd_US = icd_CA[0:3] + '.' + icd_CA[3]
                icds_US_in_corpus.append(icd_US)
                icds_CA_found.append(icd_CA)
                if icd_US in icds_US2CA.keys():
                    icds_US2CA[icd_US].extend([icd_CA])
                else:
                    icds_US2CA[icd_US] = [icd_CA]
            elif icd_CA[0:3] in all_icds_US:  # can not be found as US ICD code directly
                # for example, 4629 CA ICD code is Acute pharyngitis, acute pharyngitis, 462 US ICD code is Acute pharyngitis
                # but most of CA ICD codes in this category do not have corresponding meanings
                icd_US = icd_CA[0:3]
                icds_US_in_corpus.append(icd_US)
                icds_CA_found.append(icd_CA)
                if icd_US in icds_US2CA.keys():
                    icds_US2CA[icd_US].extend([icd_CA])
                else:
                    icds_US2CA[icd_US] = [icd_CA]
            else:
                icds_regular.append(icd_CA)
    # print(len(icds_US_in_corpus))  # we find 7718 CA ICD codes in phecode-ICD mapping files
    # print(len(icds_CA_found)) # we find 7718 CA ICD codes in phecode-ICD mapping files
    # print(len(icd_in_corpus))  # the documents contain 8539 CA ICD codes, other ~800~ ICD codes are assumed as regular word

    phecode_icd_dict_corpus = {} # get a new phecode-icd mapping,# key is phecode, value is CA ICD codes
    for i, (key, values) in enumerate(phecode_icd_dict.items()): # key is phecode, value is US ICD codes
        for icd_US in values:
            if icd_US in icds_US_in_corpus:
                icds_CA = icds_US2CA[icd_US]
                if key not in phecode_icd_dict_corpus.keys():
                    phecode_icd_dict_corpus[key] = icds_CA
                else:
                    phecode_icd_dict_corpus[key].extend(icds_CA)
    # print(len(phecode_icd_dict_corpus.keys())) # number of phecodes are 1611, number of CA ICD codes are 7718

    # tokenization for CA ICD codes and phecodes which appear in CORPUS
    mapped_phecode = {} # key is phecode, value is the mapped index of phecode from 1 to K-1, K is 1569
    for i, key in enumerate(phecode_icd_dict_corpus): # 570, key is phecode, value is mapped index
        mapped_phecode[key] = i
    mapped_icd = {} # key is icd, value is the mapped index of icd from 1 to V-1, V is 8539 as seed and regular words are included
    for i, value in enumerate(icd_in_corpus): # 8539, key is CA ICD code, value is mapped index
        mapped_icd[value] = i
    tokenized_phecode_icd = {mapped_phecode[key]: [mapped_icd[ICD_CA] for ICD_CA in value] for key, value in phecode_icd_dict_corpus.items()}
    # len(key) is 1611, len(values) is 7718

    # save phecode ICD mapping in corpus as a torch matrix
    K = len(tokenized_phecode_icd.keys())
    icd_list = mapped_icd.keys()
    V = len(icd_list)
    # print(K, V) # K is 1611, V is 8539
    seeds_topic_matrix = torch.zeros(V, K, dtype=torch.int) # 8539 x 1611
    for k, w_l in tokenized_phecode_icd.items():
        for w in w_l:
            seeds_topic_matrix[w, k] = 1
    # print(seeds_topic_matrix.sum()) # 7718 as 7718 words are seed words across topics
    torch.save(seeds_topic_matrix, "./phecode_mapping/all_seed_topic_matrix.pt")
    return mapped_phecode, mapped_icd, tokenized_phecode_icd
