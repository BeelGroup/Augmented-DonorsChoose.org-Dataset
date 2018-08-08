#!/usr/bin/env python3

import logging
import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold

import recsys

# Constant variables which might be worth reading in from a configuration file
logging.basicConfig(level=logging.INFO)
items_filepath = os.path.join('data', 'donorschoose.org', 'Donations.csv')
projects_filepath = os.path.join('data', 'donorschoose.org', 'Projects.csv')
random_state_seed = 2718281828
# Apply cleaning methods and sample the data as to reduce the amount of required memory
sampling_methods = {'user_frequency_boundary': 2, 'sample': int(1e4)}
n_jobs = 2
n_svd_components = 100
n_knn_neighbors = 40
n_nmf_components = 50
n_folds = 5

accuracy_methods = {'RMSE': recsys.rmse, 'MAE': recsys.mae}

# Implicitly rely on other commands using the current random state from numpy
np.random.seed(random_state_seed)

items = pd.read_csv(items_filepath)
projects = pd.read_csv(projects_filepath)
# Get rid of pesky whitespaces in column names
items.columns = items.columns.str.replace(' ', '')
projects.columns = projects.columns.str.replace(' ', '')

for method, opt in sampling_methods.items():
    if opt is None or opt is False:
        pass
    elif method == 'user_frequency_boundary':
        value_counts = items['DonorID'].value_counts()
        items = items[items['DonorID'].isin(value_counts.index[value_counts >= opt])]
        del value_counts
    elif method == 'sample':
        items = items.sample(n=opt)
    else:
        raise ValueError('Expected a valid sampling method, got "' + str(method) + '"')

items = pd.merge(items, projects[['ProjectID', 'SchoolID']], on='ProjectID', how='inner', sort=False)
logging.info('{:d} unique donors donated to {:d} unique projects respectively {:d} unique schools'.format(len(items['DonorID'].unique()), len(items['ProjectID'].unique()), len(items['SchoolID'].unique())))
# Convert DonationAmount into a 0/1 rating
items.at[items['DonationAmount'] > 0, 'DonationAmount'] = 1

# Create a sparse matrix for further analysis
user_ids = items['DonorID'].unique()
item_ids = items['SchoolID'].unique()
ratings = items['DonationAmount'].values
row = items['DonorID'].astype(pd.api.types.CategoricalDtype(categories=user_ids)).cat.codes
col = items['SchoolID'].astype(pd.api.types.CategoricalDtype(categories=item_ids)).cat.codes
# Utilize a Compressed Sparse Row matrix as most users merely donate once or twice
sparse_rating_matrix = csr_matrix((ratings, (row, col)), shape=(user_ids.shape[0], item_ids.shape[0]))

sparsity = 1.0 - sparse_rating_matrix.nonzero()[0].shape[0] / np.dot(*sparse_rating_matrix.shape)
logging.info('rating matrix is {:.4%} sparse'.format(sparsity))

kf = KFold(n_splits=n_folds, shuffle=True)

i = 0
algorithms = {}
algorithms['SciPy-SVD'] = recsys.SciPySVD(n_components=n_svd_components)
algorithms['SKLearn-SVD'] = recsys.SKLearnSVD(n_components=n_svd_components)
algorithms['SKLearn-KNN'] = recsys.SKLearnKNN(n_neighbors=n_knn_neighbors)
algorithms['SKLearn-NMF'] = recsys.SKLearnNMF(n_components=n_nmf_components)
# Initialize a dictionary with an entry for each algorithm which shall store accuracy values for every selected accuracy method
algorithms_error = {}
for alg_name in algorithms.keys():
    # Tuple of training error and test error for each algorithm
    algorithms_error[alg_name] = {acc_name: np.array([0., 0.]) for acc_name in accuracy_methods.keys()}

# The ordering the indices of the matrix and the user_ids, item_ids of the frame must match in order to merge the prediction back into the table
for train_idx, test_idx in kf.split(sparse_rating_matrix):
    i += 1

    for alg_name, alg in algorithms.items():
        log_line = '{:<15s} (fold {:>d}/{:<d}) ::'.format(alg_name, i, n_folds)
        train_predictions = alg.fit_transform(sparse_rating_matrix[train_idx])
        test_predictions = alg.estimate(sparse_rating_matrix[test_idx])

        for acc_name, acc in accuracy_methods.items():
            train_acc, test_acc = acc(train_predictions, sparse_rating_matrix[train_idx]), acc(test_predictions, sparse_rating_matrix[test_idx])
            algorithms_error[alg_name][acc_name] += np.array([train_acc, test_acc]) / n_folds
            log_line += ' | Training-{0:s}: {train_acc:>7.2f}, Test-{0:s}: {test_acc:>7.2f}'.format(acc_name, train_acc=train_acc, test_acc=test_acc)

        logging.info(log_line)

for alg_name, acc_methods in algorithms_error.items():
    log_line = '{:<15s} (average) ::'.format(alg_name)
    for acc_name, acc_value in acc_methods.items():
        log_line += ' | Training-{0:s}: {train_acc:>7.2f}, Test-{0:s}: {test_acc:>7.2f}'.format(acc_name, train_acc=acc_value[0], test_acc=acc_value[1])
    logging.info(log_line)
