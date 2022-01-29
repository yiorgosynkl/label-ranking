#!/usr/bin/env python3

# import os
import subprocess
from datetime import datetime
import timeit

start_time = timeit.default_timer()

program_name = ['pairwise_homemade.py', 'pairwise_modules.py', 'pairwise_modules_aggregation.py', \
                    'pairwise_kwiksort.py', 'pairwise_keep_confident.py', 'pairwise_incomplete.py'][2]
datasets =  ['algae', 'authorship', 'bodyfat', 'calhousing', 'cpu-small', 'fried', \
                'glass', 'housing', 'iris', 'pendigits', 'segment', 'stock', 'sushi_one_hot', \
                'sushi', 'vehicle', 'vowel', 'wine', 'wisconsin']

# dataset_to_ignore = { 0 , 12, 13 } # never add these                                      # all datasets (practically)
# dataset_to_ignore = { 0, 1, 2, 3, 4, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17 }               # slow datasets (est. time: 15s)
# dataset_to_ignore = { 0, 1, 5, 6, 7, 9, 11, 12, 13, 14, 16 }                              # rest and low scores 
# dataset_to_ignore = { 0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 16 }  x                     # rest (est. time: ...)
# dataset_to_ignore = { 0, 1, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17 }                  # low scores (est. time: 3h)
# dataset_to_ignore = { 0, 2, 3, 4, 5, 9, 10, 12, 13, 15, 17 }                              # quicks (est. time: 20m)
# dataset_to_ignore = { 0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17 }         # only iris (est. time: 15s)

datasets_to_check = {
    'slow_datasets' : { 5, 9, 10 },
    'medium_datasets': { 3, 4, 15, 17 },
    'quick_datasets' : { 1, 2, 6, 7, 8, 11, 14, 16 },
    'ok_datasets' : { 1, 2, 3, 4, 6, 7, 8, 11, 14, 15, 16, 17 },
    'rest': { 10, 15, 17 },
    'low_scores_datasets' : { 2, 3, 4, 6, 7, 8, 14, 16, 17 }, # plus iris and wine
    'temp' : { 10 }, # temporary
    'iris_only' : { 8 }
}['temp']


# classifiers_to_ignore = { }
classifiers_to_ignore = { 0, 1, 4, 5 }
# classifiers_to_ignore = { 0, 1, 2 }
# classifiers_to_ignore = { 0, 1, 4 }
# classifiers_to_ignore = { 0, 1 }

rows = [['index', 'dataset', 'SVC', 'DTC', 'RFC100', 'SVR', 'DTR', 'RFR100', 'time']]

print(f'running {program_name}')
for i, s in enumerate(datasets):
    if i in datasets_to_check:
        start_time_dataset = timeit.default_timer()
        row = [str(i), s]
        print(f'running dataset {s} ...')
        for c in range(6): # classifier
            if c in classifiers_to_ignore:
                output = ''
            else:
                output = subprocess.check_output(f'python3 {program_name} {i} {c}', shell=True)[:-1].decode("utf-8") 
            row.append(output)
        stop_time_dataset = timeit.default_timer()
        run_time_dataset = int(stop_time_dataset - start_time_dataset)
        row.append(f'{run_time_dataset} secs')
        rows.append(row)
    else:
        rows.append([str(i), s, ' ', ' ', ' ', ' ', ' ', ' ', ' '])

stop_time = timeit.default_timer()
run_time = int(stop_time - start_time)
print(f'Time required to create results.csv: {run_time} secs')

lf = open("results_last.csv", "w")
af = open("results_all.csv", "a")
af_header = f"""________________________________________________________________
Program name: {program_name}
Date: {str(datetime.now())[:19]}
Runtime: {run_time} secs
Info: 

"""
af.write(af_header)


for row in rows:
    line = ', '.join(row) + '\n'
    lf.write(line)
    af.write(line)
lf.close()
af.close()
