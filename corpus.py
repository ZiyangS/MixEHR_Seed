# define data structure for corpus


from typing import Mapping, List, NoReturn, Set, TypeVar
import numpy as np
import pandas as pd
import pickle
import os
import logging
import sys
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='Select one command', dest='cmd')

# parser process
parser_process = subparsers.add_parser('process', help="Transform MixEHR raw data")
parser_process.add_argument("-im", "--ignore_missing", help="Ignores observations with missing values",
                            action='store_true', default=False)
parser_process.add_argument("-n", "--max", help="Maximum number of observations to select", type=int, default=None)

# parser process
parser_split = subparsers.add_parser('split', help="Split data into train/test")
parser_split.add_argument("-tr", "--testing_rate", help="Testing rate. Default: 0.2", type=float, default=0.2)

# parser process
parser_skf = subparsers.add_parser('stratifiedcv', help="Stratified K-Folds cross-validator")
parser_skf.add_argument("-cv", "--n_splits", help="Number of folds", type=int, default=2)

# default arguments
parser.add_argument('input', help='Directory containing input data')
parser.add_argument('output', help='Directory where processed data will be stored')


