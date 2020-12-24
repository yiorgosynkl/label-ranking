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
from itertools import combinations

from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# set random seed
np.random.seed(0)

params = {                                                          # TOSET
    'csv_num' : 8 if len(sys.argv) < 2 else int(sys.argv[1]),       # choose dataset, num in {0, 1,..., 17} 
    'clf_num' : 5 if len(sys.argv) < 3 else int(sys.argv[2]),       # choose classifier, num in set {0, ..., 5}
    'antivotes_func_num' : 0                                        # choose antivotes function, num in set {0, 1, 2, 3}
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

label_names = get_labels(df.iloc[0,-1]) # names of labels (indexed from 0 to n_labels - 1)
n_labels = len(label_names)             # number of labels
feature_names = list(df.columns[:-1])   # names of features (indexed from 0 to n_features - 1)
n_features = len(feature_names)         # number of features
n_samples = len(df)                     # number of sample in dataframe

# symmetric function, converts from label-fixed ranking to position-fixed ranking and vice versa
def convert_ranking(rank):
    return np.argsort(rank)

# string -> list of nums
def convert_ranking_string2list(s):  # label-fixed expression
    return np.argsort(s.split('>'))

# # list of nums -> string
# def convert_ranking_list2string(rank, names=label_names):  # label-fixed expression
#     s = [names[ln] for ln in rank] # label number to string
#     return '>'.join(s)

# print(convert_ranking('b>c>a'))
df['ranking'] = df['ranking'].apply(convert_ranking_string2list) # change ranking for database

#----> Homemade kendall tau distance <----#

# input: 2 ranking r, l
def my_kendall_tau(l, r):
    m = len(l) # = len(l) # number of labels
    p = m * (m-1) / 2 # number of pairs of labels
    c = 0 # concordant pairs
    for a, b in combinations(range(m), 2):
        c += int((l[a] < l[b]) == (r[a] < r[b]))
    d = p - c # discordant pairs
    return (c - d) / p

# # examples
# print( my_kendall_tau([0,1,2,3], [0,1,2,3]) )
# print( my_kendall_tau([0,1,2,3], [3,2,1,0]) )
# print( my_kendall_tau([0,1,2,3], [0,3,2,1]) )

#________________ Train Binary Classifiers and Make Predictions Using Kfold ________________#

bclfs_keys = [(i,j, f'{i}-{j}') for i in range(n_labels - 1) for j in range(i+1, n_labels)]
n_bclfs = len(bclfs_keys) # n * (n-1) // 2

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


def conv(ranking, mn, mx):      # produce data for each classifier and add it to the dataframe
    return int(ranking[mn] > ranking[mx]) # mn-label is leftmost, better <-> 0. mx-label is leftmost, better, prefferred --> 1.
for i, j, mid in bclfs_keys:
    df[mid] = df['ranking'].apply(lambda r: conv(r, i, j))


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


n_folds = 5
shuffled_indices = np.random.permutation(n_samples) 
folds_test_indices = np.array_split(shuffled_indices, n_folds) # holds the test indices for each fold 
scores = [0.0]*n_folds
for f in range(n_folds):
    # bclfs = classifier_init() # initialise again to clean up model before fitting, doesn't make any difference
    test_indices = set(folds_test_indices[f])
    train_df, test_df = df[~df.index.isin(test_indices)], df[df.index.isin(test_indices)]
    x_train = train_df[feature_names]
    x_test = test_df[feature_names]
    bclfs_pred = {}
    for _, _, mid in bclfs_keys:
        y_train = train_df[mid]
        y_test = test_df[mid]
        model = bclfs[mid]
        model.fit(x_train, y_train)
        bclfs_pred[mid] = model.predict(x_test)
    # construct ranking predictions
    n_preds = len(x_test)
    y_test = test_df['ranking']
    y_pred = []
    for p in range(n_preds):
        # calculate ranking prediction from the outputs of binary classifiers
        antivotes = [0 for _ in range(n_labels)]
        for i, j, mid in bclfs_keys:
            antivotes[i] += set_antivotes(1 - bclfs_pred[mid][p])    # when close to 0, i is better, give more antivote to j
            antivotes[j] += set_antivotes(bclfs_pred[mid][p])        # when close to 1, j is better, give antivote to i
        y_pred.append(convert_ranking(np.argsort(antivotes)))           # 2 times argsort
    fold_scores = [my_kendall_tau(yt, yp) for yt, yp in zip(list(y_test), y_pred)]
    scores[f] = np.mean(fold_scores)

print(np.round(np.mean(scores),3))
# print(str(np.round(np.mean(scores),3))+'\u00B1'+str(np.round(np.std(scores),3)))
