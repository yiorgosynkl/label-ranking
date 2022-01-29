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
#from math import exp
from scipy.stats import kendalltau

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold

# set random seed
np.random.seed(0)

params = {                                                          # TOSET
    'csv_num' : 8 if len(sys.argv) < 2 else int(sys.argv[1]),       # choose dataset, num in {0, 1,..., 17} 
    'clf_num' : 5 if len(sys.argv) < 3 else int(sys.argv[2]),       # choose classifier, num in set {0, ..., 5}
    'conf' : 0.3                                                    # choose confidence threshold 0.0 to 0.5
    # 'aggregation_num': 0 if len(sys.argv) < 4 else int(sys.argv[3])
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
    if clf_num == 3:
        return {mid: SVR(gamma='scale') for _,_,mid in bclfs_keys} 
    elif clf_num == 4:
        return {mid: DecisionTreeRegressor() for _,_,mid in bclfs_keys} 
    else: # clf_num == 5:
        return {mid: RandomForestRegressor(n_estimators=100) for _,_,mid in bclfs_keys} 

bclfs = classifier_init()
conf_counter = 0

# produce data for each classifier and add it to the dataframe
def conv(ranking, mn, mx):
    return int(ranking[mn] > ranking[mx]) # mn-label is leftmost, better <-> 0. mx-label is leftmost, better, prefferred --> 1.
for i, j, mid in bclfs_keys:
    df[mid] = df['ranking'].apply(lambda r: conv(r, i, j))

def borda_rank_aggregation(rs, ws=None):
    """Aggregate complete rankings and return a single "average" ranking using borda count
    :type rs: [[ints]], list of rankings
    :type ws: [floats], list of weights
    :rtype: [ints], rank aggregation
    """
    n_labels = len(rs[0])
    if ws == None: ws = [1]*n_labels
    votes = [0 for _ in range(n_labels)]
    for rank, w in zip(rs, ws):
        for lbl, pos in enumerate(rank):
            votes[lbl] += w*(n_labels - pos) # lower ranking => more points
    pf_rank = np.argsort(votes)[::-1]
    lf_rank = list(np.argsort(pf_rank)) 
    return lf_rank

def predict_row(test_row):
    def check_row(row):
        global params
        conf = params['conf']  # confidence
        for _, _, mid in bclfs_keys:
            # if classifier really confident and not candidate opposing the classifier, do NOT aggregate (False)
            if (test_row[mid] < conf and row[mid] == 1) or (test_row[mid] > 1-conf and row[mid] == 0):
                return False
        return True 

    candidates_df['aggregate'] = candidates_df.apply(lambda row: check_row(row), axis='columns')
    # ?? what if confidence makes everyone False ??
    aggregation_df = candidates_df.loc[candidates_df['aggregate'] == True]['ranking']   # keep only the ones to aggregate
    if len(aggregation_df) == 0: # ?? what if confidence makes everyone False ??
        global conf_counter 
        conf_counter += 1
        candidates_df['votes'] = candidates_df.apply(lambda row: sum(1 - test_row[mid] if row[mid] == 0 else test_row[mid] for _, _, mid in bclfs_keys), axis='columns' )
        return candidates_df['ranking'].loc[candidates_df['votes'].argmax()]
    else:
        return borda_rank_aggregation(list(aggregation_df))

rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1234)
scores = []
for idx_train,idx_test in rkf.split(range(len(df))):
    # step 1: train classifiers
    train_df, test_df = df.loc[idx_train], df.loc[idx_test].drop(columns=[mid for _,_,mid in bclfs_keys])
    x_train, x_test = train_df[feature_names], test_df[feature_names]
    # preds_df = pd.DataFrame(index=idx_test)
    for _, _, mid in bclfs_keys:
        y_train = train_df[mid]
        model = bclfs[mid]
        model.fit(x_train, y_train)
        test_df[mid] = list(model.predict(x_test)) # mid contains the predictions for each test sample from model 'mid'

    # step 2: use result of classifiers to make predictions
    candidates_df = train_df.drop(columns=feature_names).drop_duplicates(subset=['ranking_as_string']).reset_index(drop=True)
    candidates_df['count'] =  candidates_df.apply(lambda row: len(train_df[ train_df['ranking_as_string'] == row['ranking_as_string'] ]), axis='columns' )
    candidates_df['frequency'] =  candidates_df.apply(lambda row: row['count']/len(train_df), axis='columns' )

    test_df['prediction'] = test_df.apply(lambda row: predict_row(row), axis='columns')
    test_df['kendalltau'] = test_df.apply(lambda row: kendalltau(row['ranking'], row['prediction'])[0], axis=1)
    scores.append(np.mean(test_df['kendalltau']))    

print(str(np.round(np.mean(scores),3))+'\u00B1'+str(np.round(np.std(scores),3))+f' ({conf_counter})') 
