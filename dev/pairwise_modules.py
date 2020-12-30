#!/usr/bin/env python3
################################################################
# Author            : yiorgosynkl (find me in Github: https://github.com/yiorgosynkl)
# Repository        : label-ranking
# Description       : 
#   use binary classifiers for preference of each label, then combine all results for final ranking prediction
#   then combine all results (with ad-hoc, natural way) for final ranking prediction
#   using official implementations of kendalltau and KFold
################################################################

#________________ imports ________________#

import pandas as pd
import numpy as np
import sys
from math import exp
from scipy.stats import kendalltau

from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RepeatedKFold

# set random seed
np.random.seed(0)   # to reproduce same results for random forest

params = {                                                          # TOSET
    'csv_num' : 8 if len(sys.argv) < 2 else int(sys.argv[1]),       # choose dataset, num in set {0, 1,..., 17} 
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

# # list of nums -> string
# def convert_ranking_list2string(rank, names=label_names):  # label-fixed expression
#     s = [names[ln] for ln in rank] # label number to string
#     return '>'.join(s)

# print(convert_ranking('b>c>a'))
df['ranking'] = df['ranking'].apply(convert_ranking_string2list) # change ranking for database

#________________ Train Binary Classifiers and Make Predictions Using Kfold ________________#

bclfs_keys = [(i,j, f'{i}-{j}') for i in range(n_labels - 1) for j in range(i+1, n_labels)]     # each model is given an mid (model identification)
n_bclfs = len(bclfs_keys) # n * (n-1) // 2

def classifier_init():  # choose type of binary classifier
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
    antivotes = [0 for _ in range(n_labels)]
    for i, j, mid in bclfs_keys:
        antivotes[i] += set_antivotes(1 - row[mid])   # when close to 0, i is better, give more antivote to j
        antivotes[j] += set_antivotes(row[mid])       # when close to 1, j is better, give antivote to i
    return convert_ranking(np.argsort(antivotes))     # 2 times argsort to find label-fixed ranking

rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=0)
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
    test_df['prediction'] = test_df.apply(lambda row: find_prediction(row), axis=1) # prediction based on the models' values for each test row
    test_df['kendalltau'] = test_df.apply(lambda row: kendalltau(row['ranking'], row['prediction'])[0], axis=1)
    
    scores.append(np.mean(test_df['kendalltau']))

    # TODO: with dataframes it is way to slow, probably object type is bad

print(str(np.round(np.mean(scores),3))+'\u00B1'+str(np.round(np.std(scores),3)))
