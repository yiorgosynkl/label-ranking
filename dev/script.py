#!/usr/bin/env python3

# import os
import subprocess
from datetime import datetime
import timeit

start_time = timeit.default_timer()

datasets = ['algae', 'authorship', 'bodyfat', 'calhousing', 'cpu-small', 'fried', 'glass', 'housing', 'iris', 'pendigits', 'segment', 'stock', 'sushi_one_hot', 'sushi', 'vehicle', 'vowel', 'wine', 'wisconsin']
# ignore = { 0 , 13 }
# ignore = { 0, 3, 5, 9, 12, 13 } # time-consuming datasets: 3, 5, 9, 12 | impossible datasets: 0, 13
ignore = { 0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17 } # time-consuming datasets: 3, 5, 9, 12 | impossible datasets: 0, 13

rows = [['index', 'dataset', 'SVC', 'DTC', 'RFC100', 'SVR', 'DTR', 'RFR100']]

for i, s in enumerate(datasets):
    if i in ignore:
        rows.append([str(i), s, ' ', ' ', ' ', ' ', ' ', ' '])
    else:
        row = [str(i), s]
        print(f'running dataset {s} ...')
        for c in range(6): # classifier
            output = subprocess.check_output(f'python3 pairwise.py {i} {c}', shell=True)[:-1].decode("utf-8") 
            row.append(output)
        rows.append(row)

stop_time = timeit.default_timer()
run_time = int(stop_time - start_time)
print(f'Time required to create results.csv: {run_time} secs')

lf = open("results_last.csv", "w")
af = open("results_all.csv", "a")
af_header = f"""________________________________________________________________
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
