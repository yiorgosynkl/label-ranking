# label-ranking

A repo containing the code of my thesis.

## abstract

Label ranking is a problem that has recently attracted research attention due to its generality and number of applications. Several approaches for solving the label ranking problem have been pro- posed. In this work however, we focused on the modular RPC approach that solves label ranking using pairwise comparisons and aggregation techniques. It was first presented by Hullermeier and Fürnkranz in 2008 but still remains a state-of-the-art approach, thus further investigation is required. In this work, we conducted experimental evaluation of the two-stage RPC model. We experimented with several learning algorithms at the first stage and concluded that they significantly impact the predictions of the RPC model. We also inspected popular and innovative aggregation techniques at the second stage and concluded that the results are not seriously affected, given that the aggregation techniques follow a simple and solid logical procedure. Most importantly, we interpret the scores of the model to make conclusions on why some learning algorithms and aggregation techniques work better than the rest. Lastly, we make thorough analysis on the datasets, used for benchmarking. With the use of different metrics and visualisations, we explain the reasons why label ranking is a highly complex problem. We conclude that the mapping function between the instances’ space and the rankings’ space severely affects the performance of all models and prohibits the development of a model that efficiently solves all learning problems.

> Keywords Preference learning, Label Ranking, Pairwise classification, Rank aggregation


## development

the data directory contains different datasets for training.
the dev directory contains different `.py` that create learning models and try to solve the label-ranking problem effectively. Each model uses different variations of the RPC method to achieve efficient results.


We distinguish the following teams of programs:
* programs that train models and make predictions for complete data(`pairwise_helpers.py, pairwise_kwiksort.py, pairwise_homemade.py, pairwise_modules.py, pairwise_modules_aggregation.py, pairwise_keep_confident.py`)
* programs that train models and make predictions for incomplete data(`pairwise_incomplete.py`)
* programs that make analysis of the datasets (`dataset_analysis.py, dataset_analysis2.py, scatter.py`)
* scripts to automate running the aforementioned programs (`script.py, incomplete_run.py, incomplete_run2.py`)

> Note that all python progams use global variables to control the behaviour (like "which datasets should be analysed?" or "which classifiers to be used for the models?" etc.).

## how to run

To run, activate the virtual environment, install the requirements and run the relevant scritps:
```bash
virtualenv venv 
source venv/bin/activate
python3 -m pip install -r requirements.txt

python3 scatter_analysis/scatter.py
```
