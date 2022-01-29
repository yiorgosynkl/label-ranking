#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################
# Author            : yiorgosynkl (find me in Github: https://github.com/yiorgosynkl)
# Repository        : label-ranking
# Date Created      : 20210410
# Description       : use binary classifiers for preference of each label, then combine all results for final ranking prediction
################################################################

#________________ imports ________________#

import sys
import pandas as pd
import numpy as np
from math import exp
import random
from scipy.stats import kendalltau

#from itertools import combinations, permutations
#
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RepeatedKFold

# TOSET parameters
random.seed(0)
np.random.seed(0)                                                   # set random seeds
params = {                                                          # TOSET
    'csv_num' : 8 if len(sys.argv) < 2 else int(sys.argv[1]),       # choose dataset, num in {0, 1,..., 17} 
    'clf_num' : 5 if len(sys.argv) < 3 else int(sys.argv[2]),       # choose classifier, num in set {0, ..., 5}
    'in_prob' : 0.5,                                                # incomplete prob
    'antivotes_func_num' : 0
    # 'aggregation_num': 0 if len(sys.argv) < 4 else int(sys.argv[3])
}


# import dataset
csv_choices =  ['algae', 'authorship', 'bodyfat', 'calhousing', 'cpu-small', 'fried',           \
                'glass', 'housing', 'iris', 'pendigits', 'segment', 'stock', 'sushi_one_hot',   \
                'sushi', 'vehicle', 'vowel', 'wine', 'wisconsin']
csv_num = 8 if len(sys.argv) < 2 else int(sys.argv[1])  # TOSET, choose dataset, num in {0, 1,..., 17} from above choices
CSV_PATH = f'../data/{csv_choices[csv_num]}.txt'
df = pd.read_csv(CSV_PATH)

#________________ preprocess functions ________________#

def get_labels(s):
    """Get a list of label names"""
    names = s.split('>')
    return sorted(names)

label_names = get_labels(df.iloc[0,-1])     # names of labels (indexed from 0 to n_labels - 1)
n_labels = len(label_names)                 # number of labels
feature_names = list(df.columns[:-1])       # names of features (indexed from 0 to n_features - 1)
n_features = len(feature_names)             # number of features
n_samples = len(df)                         # number of samples


# 2 ways to express a ranking: *label-fixed (lf) and *position-fixed (pf)
# label fixed means that 1st cell contains the position of label_1 
# position fixed means that 1st cell contains the label-number of the 1st position
# missing labels will have the value n_labels (because it is bigger the limits of the array)

def ranking_lf2pf(old_rank):
    """Convert a ranking from label-fixed format to position-fixed format.
    :type old_rank: [ints] (where value == n_labels the label does not exist in ranking)
    :rtype: [ints] (len of list < n_labels if ranking incomplete)
    """
    global n_labels
    new_rank = list(np.argsort(old_rank))
    mis = old_rank.count(n_labels)
    return new_rank[:len(new_rank)-mis]

def ranking_pf2lf(old_rank):
    """Convert a ranking from position-fixed format to label-fixed format.
    :type old_rank: [ints] (len of list < n_labels if ranking incomplete)
    :rtype: [ints] (where value == n_labels the label does not exist in ranking)
    """
    global n_labels
    exist = set(old_rank)                   # labels that exits will be the valid indices for new rank
    help_rank = list(np.argsort(old_rank))
    new_rank = [n_labels] * n_labels        # initialise with all missing
    k = 0
    for i in range(n_labels):
        if i in exist:
            new_rank[i] = help_rank[k]
            k += 1
    return new_rank

# def convert_string2lf(s): 
#     """Take a ranking in string format and produce incomplete (amount based on prob) rankings in label-fixed format
#     :type s: string, ranking (eg. 'c>d>b>a')
#     :type prob: float, 0 <= prob <= 1, probability that a label goes missing
#     :rtype: [ints], label-ranking format
#     """
#     global params
#     comp_rank_lf = list(np.argsort(s.split('>')))
#     comp_rank_pf = ranking_lf2pf(comp_rank_lf)
#     incomp_rank_pf = [n for n in comp_rank_pf if random.random() < 1-params['in_prob']]
#     incomp_rank_lf = ranking_pf2lf(incomp_rank_pf)
#     return incomp_rank_lf

def convert_string2lf(s): 
    """Take a ranking in string format and produce incomplete (amount based on prob) rankings in label-fixed format
    :type s: string, ranking (eg. 'c>d>b>a')
    :type prob: float, 0 <= prob <= 1, probability that a label goes missing
    :rtype: [ints], label-ranking format
    """
    global params
    comp_rank_lf = list(np.argsort(s.split('>')))
    return comp_rank_lf
    comp_rank_pf = ranking_lf2pf(comp_rank_lf)
    incomp_rank_pf = [n for n in comp_rank_pf if random.random() < 1-params['in_prob']]
    incomp_rank_lf = ranking_pf2lf(incomp_rank_pf)
    return incomp_rank_lf

#def convert_lf2incomplete(comp_rank_lf): 
#    """Take a ranking in string format and produce incomplete (amount based on prob) rankings in label-fixed format
#    :type s: string, ranking (eg. 'c>d>b>a')
#    :type prob: float, 0 <= prob <= 1, probability that a label goes missing
#    :rtype: [ints], label-ranking format
#    """
#    global params
#    comp_rank_pf = ranking_lf2pf(comp_rank_lf)
#    incomp_rank_pf = [n for n in comp_rank_pf if random.random() < 1-params['in_prob']]
#    incomp_rank_lf = ranking_pf2lf(incomp_rank_pf)
#    return incomp_rank_lf

def convert_lf2incomplete(comp_rank_lf): 
    """Take a ranking in string format and produce incomplete (amount based on prob) rankings in label-fixed format
    :type s: string, ranking (eg. 'c>d>b>a')
    :type prob: float, 0 <= prob <= 1, probability that a label goes missing
    :rtype: [ints], label-ranking format
    """
    global params
    incomp_rank_lf = [i if random.random() < 1-params['in_prob'] else -1 for i in comp_rank_lf]
    return incomp_rank_lf

# TODO: convert ranking list2string

df.rename(columns={'ranking':'string_ranking'}, inplace=True)
df['complete_ranking'] = df['string_ranking'].apply(convert_string2lf) # change ranking for database
df['incomplete_ranking'] = df['complete_ranking'].apply(convert_lf2incomplete) # change ranking for database

# define names for binary classifiers (will be used for looping)
bclfs_keys = [(i,j, f'{i}-{j}') for i in range(n_labels - 1) for j in range(i+1, n_labels)]
n_bclfs = len(bclfs_keys) # n * (n-1) // 2

# TODO: not all labels exist for comparison
# produce data for each classifier and add it to the dataframe
def rank2binary(ranking, mn, mx):
    mn_pos, mx_pos = ranking[mn], ranking[mx]
    return -1 if mn_pos == -1 or mx_pos == -1 else int(mn_pos > mx_pos) 
    # mn-label is leftmost, better <-> 0. mx-label is leftmost, better, prefferred --> 1.
for i, j, mid in bclfs_keys:
    df[mid] = df['incomplete_ranking'].apply(lambda r: rank2binary(r, i, j))

# create classifiers
bclfs = {                                                           # choose type of binary classifier
  0: {mid: SVC(gamma='scale') for _,_,mid in bclfs_keys},           # binary classifiers (select by using model id)
  1: {mid: DecisionTreeClassifier() for _,_,mid in bclfs_keys},
  2: {mid: RandomForestClassifier(n_estimators=100) for _,_,mid in bclfs_keys}, 
  3: {mid: SVR(gamma='scale') for _,_,mid in bclfs_keys}, 
  4: {mid: DecisionTreeRegressor() for _,_,mid in bclfs_keys}, 
  5: {mid: RandomForestRegressor(n_estimators=100) for _,_,mid in bclfs_keys}
}.get(params['clf_num'], {mid: RandomForestRegressor(n_estimators=100) for _,_,mid in bclfs_keys}) # set default

#________________ Train Binary Classifiers and Make Predictions Using Kfold ________________#

def set_antivotes(conf):    # when conf -> 1, antivotes -> 0 , should be decreasing monotonic function
    global params
    antivotes_func_num = params['antivotes_func_num']
    if antivotes_func_num == 1:
        return (1-conf)**2          # 0.0 to 1.0 -> [1.0, 0.81, 0.64, 0.49, 0.36, 0.25, 0.16, 0.9, 0.4, 0.1, 0.0]
    elif antivotes_func_num == 2:
        return min(1/conf if conf > 0 else 50, 50)  # 0.0 to 1.0 -> [50, 10.0, 5.0, 3.33, 2.5, 2.0, 1.66, 1.43, 1.25, 1.11, 1.0]
    elif antivotes_func_num == 3:
        return 1/exp(4*conf)        # 0.0 to 1.0 -> [1.0, 0.67, 0.45, 0.30, 0.20, 0.14, 0.09, 0.06, 0.04, 0.03, 0.02]
    else:
        return 1-conf               # 0.0 to 1.0 -> [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

def find_prediction(row):
    global n_labels
    antivotes = [0 for _ in range(n_labels)]
    for i, j, mid in bclfs_keys:
        antivotes[i] += set_antivotes(1 - row[mid])   # when close to 0, i is better, give more antivote to j
        antivotes[j] += set_antivotes(row[mid])       # when close to 1, j is better, give antivote to i
    return np.argsort(np.argsort(antivotes))          # 2 times argsort to find label-fixed ranking

rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1234)
scores = []
for fold_train_idxs, fold_test_idxs in rkf.split(range(len(df))):
    # step 1: train classifiers
    test_df = df.loc[fold_test_idxs].drop(columns=[mid for _,_,mid in bclfs_keys])
    x_test = test_df[feature_names]
    for _, _, mid in bclfs_keys:
        model_train_idxs = df.index[df[mid] != -1]
        valid_train_idxs = np.intersect1d(fold_train_idxs, model_train_idxs) # training with partial data
        train_df = df.loc[valid_train_idxs]
        x_train, y_train = train_df[feature_names], train_df[mid]
        model = bclfs[mid]
        model.fit(x_train, y_train)
        test_df[mid] = list(model.predict(x_test)) # mid contains the predictions for each test sample from model 'mid'
    # # step 2: use result of classifiers to make predictions
    test_df['prediction'] = test_df.apply(lambda row: find_prediction(row), axis=1) # full prediction based on the models' values for each test row
    test_df['kendalltau'] = test_df.apply(lambda row: kendalltau(row['complete_ranking'], row['prediction'])[0], axis=1)    
    scores.append(np.mean(test_df['kendalltau']))

print(str(np.round(np.mean(scores),3))+'\u00B1'+str(np.round(np.std(scores),3)))
