#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################
# Author            : yiorgosynkl (find me in Github: https://github.com/yiorgosynkl)
# Repository        : label-ranking
# Date Created      : 20201126
# Description       : use binary classifiers for preference of each label, then combine all results for final ranking prediction
################################################################

#________________ imports ________________#

import pandas as pd
import numpy as np
import sys
from math import exp
from itertools import combinations, permutations
from scipy.stats import kendalltau

from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RepeatedKFold

# set random seed
# np.random.seed(0)

params = {                                                          # TOSET
    'csv_num' : 8 if len(sys.argv) < 2 else int(sys.argv[1]),       # choose dataset, num in {0, 1,..., 17} 
    'clf_num' : 0 if len(sys.argv) < 3 else int(sys.argv[2]),       # choose classifier, num in set {0, ..., 5}
    'normalise_num' : 0                                             # choose normalise functions, num in set {0, 1, 2, 3}
}

# import dataset
csv_choices = ['algae', 'authorship', 'bodyfat', 'calhousing', 'cpu-small', 'fried', 'glass', 'housing', 'iris', 'pendigits', 'segment', 'stock', 'sushi_one_hot', 'sushi', 'vehicle', 'vowel', 'wine', 'wisconsin']
csv_num = params['csv_num']
CSV_PATH = f'../data/{csv_choices[csv_num]}.txt'  # path to dataset
df = pd.read_csv(CSV_PATH)

#________________ preprocess functions ________________#


# s string -> return n_labels and label_names
def get_labels(s):
  names = s.split('>')
  return sorted(names)

# general dimensions and information
label_names = get_labels(df.iloc[0,-1]) # names of labels (indexed from 0 to n_labels - 1)
n_labels = len(label_names)             # number of labels
feature_names = list(df.columns[:-1])   # names of features (indexed from 0 to n_features - 1)
n_features = len(feature_names)         # number of features
n_samples = len(df)

# symmetric function, converts from lf to pf and from pf to lf
def convert_ranking(rank):
    return np.argsort(rank)

# string -> list of nums
def convert_ranking_string2list(s):  # label-fixed expression
    return np.argsort(s.split('>'))

# list of nums -> string
def convert_ranking_list2string(rank, names=label_names):  # label-fixed expression
    s = [names[ln] for ln in rank] # label number to string
    return '>'.join(s)

# print(convert_ranking('b>c>a'))
df['ranking'] = df['ranking'].apply(convert_ranking_string2list) # change ranking for database

#----> Homemade Borda Count for complete rankings <----#

# list of rankings
def borda_count(l):
    n_ranks, n_labels = len(l), len(l[0])
    # points equal to position => more points means lower ranking
    counts = [sum(l[i][j] for i in range(n_ranks)) for j in range(n_labels)]
    rank = [0 for i in range(n_labels)]
    for idx, val in zip(np.argsort(counts), range(n_labels)):
        rank[idx] = val
    return rank

# # examples
# print(borda_count([[3,0,1,2], [3,1,0,2], [2,0,1,3]]))

######################################################################
#________________ Binary Classifiers ________________#

bclfs_keys = [(i,j, f'{i}-{j}') for i in range(n_labels - 1) for j in range(i+1, n_labels)]
n_bclfs = len(bclfs_keys) # n * (n-1) // 2

# choose type of binary classifier
def classifier_init():
    global params
    clf_num = params['clf_num']
    if clf_num == 0: 
        return {mid: SVC(gamma='scale') for _,_,mid in bclfs_keys} # binary classifiers (select by using model id)
    elif clf_num == 1:
        return {mid: DecisionTreeClassifier() for _,_,mid in bclfs_keys} 
    elif clf_num == 2:
        return {mid: RandomForestClassifier(n_estimators=100) for _,_,mid in bclfs_keys} 
    elif clf_num == 3:
        return {mid: SVR(gamma='scale') for _,_,mid in bclfs_keys} 
    elif clf_num == 4:
        return {mid: DecisionTreeRegressor() for _,_,mid in bclfs_keys} 
    else: # clf_num == 5:
        return {mid: RandomForestRegressor(n_estimators=100) for _,_,mid in bclfs_keys} 

bclfs = classifier_init()

# print(bclfs)

# produce data for each classifier and add it to the dataframe
def conv(ranking, mn, mx):
    return int(ranking[mn] > ranking[mx]) # mn-label is leftmost, better <-> 0. mx-label is leftmost, better, prefferred --> 1.
for i, j, mid in bclfs_keys:
    df[mid] = df['ranking'].apply(lambda r: conv(r, i, j))

#________________ Binary classifiers, outer loop folds, inner loops models ________________#

def antis(p): # when p -> 1, antivotes -> 0 , should be decreasing monotonic function
    global params
    normalise_num = params['normalise_num'] # choose normalise functions, num in set {0, 1, 2, 3} from choices below
    if normalise_num == 1:
        return (1-p)**2  # 0.0 to 1.0 -> [1.0, 0.81, 0.64, 0.49, 0.36, 0.25, 0.16, 0.9, 0.4, 0.1, 0.0]
    if normalise_num == 2:
        return min(1/p if p > 0 else 50, 50) # 0.0 to 1.0 -> [50, 10.0, 5.0, 3.33, 2.5, 2.0, 1.66, 1.43, 1.25, 1.11, 1.0]
    if normalise_num == 3:
        return 1/exp(4*p)  # 0.0 to 1.0 -> [1.0, 0.67, 0.45, 0.30, 0.20, 0.14, 0.09, 0.06, 0.04, 0.03, 0.02]
    return 1-p

rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=0)
scores = []
for idx_train,idx_test in rkf.split(range(len(df))):
    train_df, test_df = df.loc[idx_train], df.loc[idx_test]
    x_train, x_test = train_df[feature_names], test_df[feature_names]
    bclfs_pred = {}
    for _, _, mid in bclfs_keys:
        y_train, y_test = train_df[mid], test_df[mid]
        model = bclfs[mid]
        model.fit(x_train, y_train)
        bclfs_pred[mid] = model.predict(x_test)
    # construct ranking predictions
    n_preds = len(x_test)
    antivotes = [[0 for _ in range(n_labels)] for _ in range(n_preds)]
    for i, j, mid in bclfs_keys:
        for p in range(n_preds):
            antivotes[p][i] += antis(1 - bclfs_pred[mid][p]) # when close to 0, i is better, give more antivote to j
            antivotes[p][j] += antis(bclfs_pred[mid][p]) # when close to 1, j is better, give antivote to i
    y_test = test_df['ranking']
    y_pred = [convert_ranking(np.argsort(p)) for p in antivotes]
    fold_scores = [kendalltau(yt, yp)[0] for yt, yp in zip(list(y_test), y_pred)]
    scores.append(np.average(fold_scores))

# print(scores)
# print(f'score: {np.average(scores)}')
print(np.average(scores))

# TODOS: mean and variance
