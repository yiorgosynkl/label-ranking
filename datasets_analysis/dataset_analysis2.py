#!/usr/bin/env python3
################################################################
# Author            : yiorgosynkl (find me in Github: https://github.com/yiorgosynkl)
# Repository        : label-ranking
# Description       : datasets analysis 2
################################################################

'''
    Extensive Description

    This program plots the amount of unique rankings that we encounter in a dataset.
    It gives an idea on the distribution of rankings in the input space of the problem.

    We expect that datasets where only a minority of the unique rankings appear the majority of the times 
    are also datasets that RPC performs better.
    
    This will help us find corellation between the complexity in the input space and the performance of our models.
'''


#________________ imports ________________#

import pandas as pd
import numpy as np
from scipy.stats import kendalltau
from math import factorial

import timeit
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# set random seed
np.random.seed(0)   # to reproduce same results for random forest

# params = {                                                          # TOSET
#     'clf_num' : 2 if len(sys.argv) < 3 else int(sys.argv[2]),       # choose classifier, num in set {0, ..., 5}
# }


# import dataset
# csv_choices = ['authorship', 'bodyfat', 'calhousing', 'cpu-small', 'fried', 'glass', 'housing', 'iris', 'pendigits', 'segment', 'stock', 'vehicle', 'vowel', 'wine', 'wisconsin']
# csv_choices = ['fried', 'pendigits', 'segment', 'vowel', 'wisconsin']
# csv_choices = ['authorship', 'bodyfat', 'calhousing', 'cpu-small', 'glass', 'housing', 'iris', 'stock', 'vehicle', 'vowel', 'wine', 'wisconsin']
# csv_choices = ['iris', 'wine']
csv_choices = ['fried']

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

    df['ranking_as_string'] = df['ranking']
    df['ranking'] = df['ranking_as_string'].apply(convert_ranking_string2list) # change ranking for database

    unique_df = df.drop_duplicates(subset=['ranking_as_string']).reset_index(drop=True)  # hold only uniques

    unique_rankings = unique_df['ranking']
    unique_rankings_as_string = unique_df['ranking_as_string']
    n_unique_rankings = len(unique_rankings)

    print(f'--------> {csv_name} (unique rankings: {n_unique_rankings}) <--------')
    print( 3 * ' , ' + 'ranking as string, ' + ','.join(list(unique_rankings_as_string)))
    print( 'ranking as list, ranking as string, count, index,' + ','.join(str(i) for i in range(n_unique_rankings)))

    out_array = [[0.0 for _ in range(n_unique_rankings)] for _ in range(n_unique_rankings)]
    for i in range(n_unique_rankings):
        for j in range(i, n_unique_rankings):
            out_array[i][j] = out_array[j][i] = np.round( kendalltau(unique_rankings[i],unique_rankings[j])[0], 3)
        
    counts = []
    for i, row in enumerate(out_array):
        count = len(df[ df['ranking_as_string'] == unique_rankings_as_string[i] ])
        counts.append(count)
        out_string = f'{unique_rankings[i]}, {unique_rankings_as_string[i]}, {count}, {i}, ' + ','.join(str(x) for x in row)
        print(out_string)

    counts.sort()

    # plt.figure()
    fig, ax = plt.subplots()    
    plt.plot(range(len(counts)), counts)
    ax.set(title=f'sorted number of appearances of unique rankings\n(0 appearances not included)\ndataset: {csv_name}', ylabel='number of appereances')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(f'dataset_analysis/{csv_name}_classifier_appereances_no0.png')
    plt.close(fig)

    n_total_rankings = factorial(n_labels)
    if n_total_rankings < 10**7:
        n_missing_rankings = n_total_rankings - len(counts)
        counts = [0]*n_missing_rankings + counts

        # plt.figure()
        fig, ax = plt.subplots()
        plt.plot(range(len(counts)), counts)
        ax.set(title=f'sorted number of appearances of unique rankings\n(0 appearances included)\ndataset: {csv_name}', ylabel='number of appereances')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(f'dataset_analysis/{csv_name}_classifier_appereances_with0.png')
        plt.close(fig)

