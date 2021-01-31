#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################
# Author            : yiorgosynkl (find me in Github: https://github.com/yiorgosynkl)
# Repository        : label-ranking
################################################################

#________________ imports ________________#

import pandas as pd
import numpy as np
import sys
from scipy.stats import kendalltau

from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RepeatedKFold

# set random seed
np.random.seed(0)

params = {                                                          # TOSET
    'csv_num' : 8 if len(sys.argv) < 2 else int(sys.argv[1]),       # choose dataset, num in {0, 1,..., 17} 
    'clf_num' : 2 if len(sys.argv) < 3 else int(sys.argv[2]),       # choose classifier, num in set {0, ..., 5}
    # 'aggregation_num': 9 if len(sys.argv) < 4 else int(sys.argv[3])
}

# import dataset
csv_choices = ['algae', 'authorship', 'bodyfat', 'calhousing', 'cpu-small', 'fried', \
                'glass', 'housing', 'iris', 'pendigits', 'segment', 'stock', 'sushi_one_hot', \
                'sushi', 'vehicle', 'vowel', 'wine', 'wisconsin']
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

df['ranking_as_string'] = df['ranking']
df['ranking'] = df['ranking_as_string'].apply(convert_ranking_string2list) # change ranking for database

#________________ Train Binary Classifiers and Make Predictions Using Kfold ________________#

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

# produce data for each classifier and add it to the dataframe
def conv(ranking, mn, mx):
    return int(ranking[mn] > ranking[mx]) # mn-label is leftmost, better <-> 0. mx-label is leftmost, better, prefferred --> 1.
for i, j, mid in bclfs_keys:
    df[mid] = df['ranking'].apply(lambda r: conv(r, i, j))

def kwiksort(row):
    """Given binary preference between labels, 
    construct the relevant majority tournament and implement kwiksort
    """
    # produce adjeceny matrix of tournament (weights)
    adj_matrix = [[0 for _ in range(n_labels)] for _ in range(n_labels)]  # 1 if edge from row node to col node
    for i, j, mid in bclfs_keys:    # for each pair
        if row[mid] <= 0.5:     # when close to 0, i is better,
            adj_matrix[i][j] = 1
        else:
            adj_matrix[j][i] = 1

    def recurse(arr):
        if len(arr) <= 1:
            return arr
        slow, fast = 0, 0 # slow and fast pointer and pivot = arr[-1] 
        while fast < len(arr) - 1:
            start_node, end_node = arr[fast], arr[-1]
            if adj_matrix[start_node][end_node] == 1: 
                arr[fast], arr[slow] = arr[slow], arr[fast]
                slow += 1
            fast += 1
        arr[slow], arr[-1] = arr[-1], arr[slow]
        return recurse(arr[:slow]) + [ arr[slow] ] + recurse(arr[slow+1:])
    
    begin_pf = [i for i in range(n_labels)]     # TODO: shuffle, loop many times and aggregate 
    aggregate_pf = recurse(begin_pf)
    aggregate_lf = list(np.argsort(aggregate_pf))
    return aggregate_lf

rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1234)
scores = []
for idx_train,idx_test in rkf.split(range(len(df))):
    # step 1: train classifiers
    train_df, test_df = df.loc[idx_train], df.loc[idx_test].drop(columns=[mid for _,_,mid in bclfs_keys])
    x_train, x_test = train_df[feature_names], test_df[feature_names]
    for _, _, mid in bclfs_keys:
        y_train = train_df[mid]
        model = bclfs[mid]
        model.fit(x_train, y_train)
        test_df[mid] = list(model.predict(x_test)) # mid contains the predictions for each test sample from model 'mid'

    # step 2: use result of classifiers to make predictions
    test_df['prediction'] = test_df.apply(lambda row: kwiksort(row), axis=1) 
    test_df['kendalltau'] = test_df.apply(lambda row: kendalltau(row['ranking'], row['prediction'])[0], axis=1)
    scores.append(np.mean(test_df['kendalltau']))

print(str(np.round(np.mean(scores),3))+'\u00B1'+str(np.round(np.std(scores),3)))
