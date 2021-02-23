#!/usr/bin/env python3
################################################################
# Author            : yiorgosynkl (find me in Github: https://github.com/yiorgosynkl)
# Repository        : label-ranking
# Description       : datasets analysis
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
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

import timeit
import matplotlib.pyplot as plt

# set random seed
np.random.seed(0)   # to reproduce same results for random forest

# params = {                                                          # TOSET
#     'clf_num' : 2 if len(sys.argv) < 3 else int(sys.argv[2]),       # choose classifier, num in set {0, ..., 5}
# }


# import dataset
# csv_choices = ['authorship', 'bodyfat', 'calhousing', 'cpu-small', 'fried', 'glass', 'housing', 'iris', 'pendigits', 'segment', 'stock', 'vehicle', 'vowel', 'wine', 'wisconsin']
# csv_choices = ['fried', 'pendigits', 'segment', 'vowel', 'wisconsin']
# csv_choices = ['authorship', 'bodyfat', 'calhousing', 'cpu-small', 'glass', 'housing', 'iris', 'stock', 'vehicle', 'vowel', 'wine', 'wisconsin']
csv_choices = ['iris']

# s string -> return n_labels and label_names
def get_labels(s):
    names = s.split('>')
    return sorted(names)

# symmetric function, converts from lf to pf and from pf to lf
def convert_ranking(rank):
    return np.argsort(rank)

# string -> list of nums
def convert_ranking_string2list(s):  # label-fixed expression
    return np.argsort(s.split('>'))

print('dataset, samples, features , labels, unique rankings, auc_scores, f1_scores, accuracy_score, runtime')
for csv_name in csv_choices:
    start_time = timeit.default_timer()

    CSV_PATH = f'../data/{csv_name}.txt'  # path to dataset
    df = pd.read_csv(CSV_PATH)

    # general dimensions and information
    label_names = get_labels(df.iloc[0,-1]) # names of labels (indexed from 0 to n_labels - 1)
    n_labels = len(label_names)             # number of labels
    feature_names = list(df.columns[:-1])   # names of features (indexed from 0 to n_features - 1)
    n_features = len(feature_names)         # number of features
    n_samples = len(df)
    n_unique_rankings = len(pd.unique(df['ranking']))

    out_string = f'{csv_name}, {n_samples}, {n_features}, {n_labels}, {n_unique_rankings}, '

    df['ranking'] = df['ranking'].apply(convert_ranking_string2list) # change ranking for database

    bclfs_keys = [(i,j, f'{i}-{j}') for i in range(n_labels - 1) for j in range(i+1, n_labels)]     # each model is given an mid (model identification)
    n_bclfs = len(bclfs_keys) # n * (n-1) // 2

    def classifier_init():  # choose type of binary classifier
        return {mid: RandomForestClassifier(n_estimators=100) for _,_,mid in bclfs_keys} 
        # global params
        # clf_num = params['clf_num']
        # if clf_num == 0: 
        #     return {mid: SVC(gamma='scale') for _,_,mid in bclfs_keys} # binary classifiers (select by using model id)
        # elif clf_num == 1:
        #     return {mid: DecisionTreeClassifier() for _,_,mid in bclfs_keys} 
        # elif clf_num == 2:
        #     return {mid: RandomForestClassifier(n_estimators=100) for _,_,mid in bclfs_keys} 
        # elif clf_num == 3:
        #     return {mid: SVR(gamma='scale') for _,_,mid in bclfs_keys} 
        # elif clf_num == 4:
        #     return {mid: DecisionTreeRegressor() for _,_,mid in bclfs_keys} 
        # else: # clf_num == 5:
        #     return {mid: RandomForestRegressor(n_estimators=100) for _,_,mid in bclfs_keys} 

    bclfs = classifier_init()

    # produce data for each classifier and add it to the dataframe
    def conv(ranking, mn, mx):
        return int(ranking[mn] > ranking[mx]) # mn-label is leftmost, better <-> 0. mx-label is leftmost, better, prefferred --> 1.
    for i, j, mid in bclfs_keys:
        df[mid] = df['ranking'].apply(lambda r: conv(r, i, j))

    rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1234)
    auc_scores, f1_scores, acc_scores = [], [], []
    for idx_train,idx_test in rkf.split(range(len(df))):
        # step 1: train classifiers
        train_df, test_df = df.loc[idx_train], df.loc[idx_test]
        x_train, x_test = train_df[feature_names], test_df[feature_names]
        for _, _, mid in bclfs_keys:
            y_train = train_df[mid]
            model = bclfs[mid]
            model.fit(x_train, y_train)
            y_test = list(test_df[mid])
            y_pred = list(model.predict(x_test))
            if sum(y_test) != 0 and sum(y_test) != len(y_test):  # not all 0, not all 1
                auc_scores.append(roc_auc_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred))
            acc_scores.append(accuracy_score(y_test, y_pred))

    stop_time = timeit.default_timer()
    run_time = int(stop_time - start_time)

    auc_string_mean = f'{np.round(np.mean(auc_scores),3)}\u00B1{np.round(np.std(auc_scores),3)}'
    f1_string_mean = f'{np.round(np.mean(f1_scores),3)}\u00B1{np.round(np.std(f1_scores),3)}'
    acc_string_mean = f'{np.round(np.mean(acc_scores),3)}\u00B1{np.round(np.std(acc_scores),3)}'
    

    fig, ax = plt.subplots()    
    plt.hist(auc_scores, int(1/0.02), range=[0.0, 1.0])  # 50 bars
    ax.set(title=f'histogram of auc scores (50 bars) ({auc_string_mean})\ndataset: {csv_name}', ylabel='number of classifiers', xlabel='roc auc score')
    plt.savefig(f'dataset_analysis/{csv_name}_50bars_auc.png')
    plt.close(fig)

    fig, ax = plt.subplots()    
    plt.hist(f1_scores, int(1/0.02), range=[0.0, 1.0])  # 50 bars
    ax.set(title=f'histogram of f1 scores (50 bars) ({f1_string_mean})\ndataset: {csv_name}', ylabel='number of classifiers', xlabel='f1 score')
    plt.savefig(f'dataset_analysis/{csv_name}_50bars_f1.png')
    plt.close(fig)

    fig, ax = plt.subplots()    
    plt.hist(acc_scores, int(1/0.02), range=[0.0, 1.0])  # 50 bars
    ax.set(title=f'histogram of accuracy scores (50 bars) ({acc_string_mean})\ndataset: {csv_name}', ylabel='number of classifiers', xlabel='accuracy score')
    plt.savefig(f'dataset_analysis/{csv_name}_50bars_accuracy.png')
    plt.close(fig)

    # # plt.figure()
    # fig, ax = plt.subplots()    
    # plt.hist(scores, int(1/0.02))  # 50 bars, zoomed
    # ax.set(title=f'histogram of auc scores (50 bars, zoomed) ({auc_string_mean})', ylabel='frequency', xlabel='roc auc score')
    # plt.savefig(f'dataset_analysis/{csv_name}_50bars_zoomed.png')
    
    # # plt.figure()
    # fig, ax = plt.subplots()    
    # plt.hist(auc_scores, int(1/0.005), range=[0.0, 1.0])  # 200 bars
    # ax.set(title=f'histogram of auc scores (200 bars) ({auc_string_mean})\ndataset: {csv_name}', ylabel='number of classifiers', xlabel='roc auc score')
    # plt.savefig(f'dataset_analysis/{csv_name}_200bars_auc.png')  
    # plt.close(fig)

    # fig, ax = plt.subplots()    
    # plt.hist(f1_scores, int(1/0.005), range=[0.0, 1.0])  # 200 bars
    # ax.set(title=f'histogram of f1 scores (200 bars) ({f1_string_mean})\ndataset: {csv_name}', ylabel='number of classifiers', xlabel='f1 score')
    # plt.savefig(f'dataset_analysis/{csv_name}_200bars_f1.png')  
    # plt.close(fig)

    # fig, ax = plt.subplots()    
    # plt.hist(acc_scores, int(1/0.005), range=[0.0, 1.0])  # 200 bars
    # ax.set(title=f'histogram of accuracy scores (200 bars) ({acc_string_mean})\ndataset: {csv_name}', ylabel='number of classifiers', xlabel='accuracy score')
    # plt.savefig(f'dataset_analysis/{csv_name}_200bars_accuracy.png')  
    # plt.close(fig)

    # # plt.figure()
    # fig, ax = plt.subplots()    
    # plt.hist(scores, int(1/0.005))  # 200 bars, zoomed
    # ax.set(title=f'histogram of auc scores (200 bars, zoomed) ({auc_string_mean})', ylabel='frequency', xlabel='roc auc score')
    # plt.savefig(f'dataset_analysis/{csv_name}_200bars_zoomed.png') 

    out_string += f'{auc_string_mean}, {f1_string_mean}, {acc_string_mean}, {run_time}s'
    print(out_string)
