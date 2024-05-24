# Sub-SpaCE
This repository contains the code of "Sub-SpaCE: Subsequence-based Sparse Counterfactual Explanations for Time Series Classification Problems".

To reproduce the experiments, run the notebook `./experiments/sota_methods_ng_wcf.ipynb` to obtain the results of Native Guide and 
Watcher et al. Then run `./experiments/ab_cf.py` and `./experiments/subspace.py` to get the results of AB-CF and our method, Sub-SpaCE.

The generated counterfactuals are stored in `./experiments/results`. The metrics, as well as the tables and figures can 
be reproduced by running `./experiments/evaluation/result_tables_and_visualizations.ipynb`.

The implementation of Sub-SpaCE can be found in `./methods/SubSpaCE.py`.