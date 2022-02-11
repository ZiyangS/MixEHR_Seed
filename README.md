***Note:The author of KG-ETM submitted a wrong Anoymous link. They are sorry for the mistake. To KG-ETM please go https://anonymous.4open.science/r/Knowledge_graph-ETM-FD6F/README.md.



# MixEHR-Seed


MixEHR-Seed is a seed-guided Bayesian topic model that can fit large-scale longitundinal heterogeneous EHR data and thousands of phenotypes. 
In the seed-guidance, each topic is represented as two distributions: (1)thea seed-topic distribution over only the seed words;
(2) the regular-topic distribution over the entire vocabulary.
Moreover, by associating each patient (with a certain age) with an age-dependent topic hyperparameters, we can model temporal topic progression in the population. 
We generalize our model to multi-modality by the incorporation of diverse types of EHR data.
To learn our model, we devise a hybrid Bayesian inference algorithm in a stochastic manner. We infer the seed-guidance topic assignments by collapsed variational mean-field inference.
We also infer the age-dependent topic hyperparameters by an amortized inference using a LSTM network. 




# Relevant Publications

This published code is referenced from the submitted paper: Automatic Phenotyping by a Seed-guided Topic Model


# Dataset

We evaluated MixEHR-S on the extracted clinical dataset from the PopHR database. 
For these datasets, we consider each patient's records within a certain age group, a EHR code (such as ICD code), the patient's age, and phenotype as
document, word, time label, and topic, respectively. 


We also requires a fact table to perform prediction, in which each row and column represents a patient and a target phenotype. 


# Code Description

## STEP 1: Process Dataset

The input data file need to be processed into built-in data structure "Corpus". You can use "MixEHR_Seed/corpus.py" to process dataset and generate a runnable data structure 
for MixEHR-Seed.
Place dataset to specific path "MixEHR_Seed/data/" or edit the corresponding code to change the path. You can run following code:

    run(parser.parse_args(['process', '-n', '150', './data/', './store/']))
    
you also need to split the dataset into train/validation/test subset. The data path and detailed split ratio could be edited:
    
    run(parser.parse_args(['split', 'store/test/', 'store/']))
	
	
## STEP 2: Extraction of Seed words

To employ seed-guidance, we have to extract seed sets for topics. In our setup, we use the PheWAS PheCode-ICD code mapping to constrcut seed sets for phenotypes. 
Instead, you can build guided information by yourself.
When we run the file "MixEHR_Seed/corpus.py", the code will build the seed sets using the function in "MixEHR_Seed/utils.py".
The utilized phecode-ICD code mapping is at path "MixEHR_Seed/phecode_mapping/". 
To the end, we obtain the extracted seed set at "MixEHR_Seed/phecode_mapping/all_seed_topic_matrix.pt", where each row and column represents a word and a topic, respectively.


## STEP 3: Topic Modelling

After process dataset and extract seed words, you can run "MixEHR_Seed/main.py" to perform seed-guided topic modelling on the train set. 
The execution code is:

    run(parser.parse_args(['./test_store/', './result/']))
    

## STEP 4: Evalutions

With the saved parameters stored in training stage, we use the inferred regular topics, age-dependent topic hyperparameters, the phenotype topic mixture membership
for the evaluations of topic interpretability, temporal disease progression, and phenotype prediction. 
    
## STEP 5: Hyperparameter Tuning

For MixEHR-Seed, the topic hyperparameters of regular topics and seed topics need to fine-tune by minimizing the held-out negative log-likelihood on the validation set. 
After that, we apply MixEHR-Seed with the estiated hyperparameters on the train set.

## STEP 6: Prepare Your Own Dataset

Your prepared data should have two files: data.csv and time.csv.
- data.csv: the patient ID,  the word (such as a ICD code), the age at which record this word, the document id (a patient with a certain age), word frequency.

                            Headers:
							pat_id,icd,age_at_diagnosis,doc_id,freq


- time.csv: the patient ID, the age group of patient, the document id (a patient with a certain age) 

                            Headers:	
							pat_id,age_at_diagnosis,doc_id
