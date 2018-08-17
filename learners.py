#!/usr/bin/env python3

import logging
import os

import numpy as np
import pandas as pd
import surprise as spl
import yaml
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold
from surprise.model_selection import KFold as spl_KFold
from surprise.reader import Reader as spl_Reader

import recsys

with open('config.yml', 'r') as stream:
    config = yaml.load(stream)

# Read in all the configuration once instead of possibly failing later because of missing values
log_level = config['log_level']
donations_filepath = config['donations_filepath']
projects_filepath = config['projects_filepath']
random_state_seed = config['random_state_seed']
sampling_methods = config['sampling_methods']
n_folds = config['n_folds']
algorithms_args = config['algorithms_args']
rating_scores = config['rating_scores']
rating_range_quantile = config['rating_range_quantile']

logging.basicConfig(level=log_level)
accuracy_methods = {'RMSE': recsys.rmse, 'MAE': recsys.mae}
# Implicitly rely on other commands using the current random state from numpy
np.random.seed(random_state_seed)

donations = pd.read_csv(donations_filepath)
projects = pd.read_csv(projects_filepath)
# Get rid of pesky whitespaces in column names (pandas' query convenience function e.g. is allergic to them)
donations.columns = donations.columns.str.replace(' ', '')
projects.columns = projects.columns.str.replace(' ', '')

# Sum up duplicate donations unconditionally; Hence do not try to predict projects which the donor has already donated to
items = donations.groupby(['DonorID', 'ProjectID'])['DonationAmount'].sum().reset_index()

# Apply the cleaning and sampling operations in a fixed order independently of the order of the dict or the user's choice
sampling_methods_priority = {'drop_raw_values': 200, 'frequency_boundaries': 500, 'sample': 900}
for method, opt in sorted(sampling_methods.items(), key=lambda x: sampling_methods_priority[x[0]]):
    if opt is None or opt is False:
        pass
    elif method == 'drop_raw_values':
        for drop_query in opt:
            items = items.drop(items.query(drop_query).index)
    elif method == 'frequency_boundaries':
        # Ensure all conditions are met at the same time; Select the intersection
        sel = True
        for column, bound in opt:
            value_counts = items[column].value_counts()
            sel = sel & items[column].isin(value_counts.index[value_counts >= bound])

        items = items[sel]
    elif method == 'sample':
        items = items.sample(n=opt)
    else:
        raise ValueError('Expected a valid sampling method from ' + str(sampling_methods_priority.keys()) + ', got "' + str(method) + '"')

logging.debug('{:d} unique donors donated to {:d} unique projects'.format(items['DonorID'].unique().shape[0], items['ProjectID'].unique().shape[0]))
# Convert DonationAmount into a rating
rating_bins = np.logspace(*np.log10(items['DonationAmount'].quantile(rating_range_quantile).values), num=len(rating_scores) + 1)
rating_bins[0], rating_bins[-1] = items['DonationAmount'].min(), items['DonationAmount'].max()
items['DonationAmount'] = pd.cut(items['DonationAmount'], bins=rating_bins, include_lowest=True, labels=rating_scores, retbins=False).astype(np.float)

# Create a sparse matrix for further analysis
user_ids = items['DonorID'].unique()
item_ids = items['ProjectID'].unique()
ratings = items['DonationAmount'].values
row = items['DonorID'].astype(pd.api.types.CategoricalDtype(categories=user_ids)).cat.codes
col = items['ProjectID'].astype(pd.api.types.CategoricalDtype(categories=item_ids)).cat.codes
# Utilize a Compressed Sparse Row matrix as most users merely donate once or twice
sparse_rating_matrix = csr_matrix((ratings, (row, col)), shape=(user_ids.shape[0], item_ids.shape[0]))

sparsity = 1.0 - sparse_rating_matrix.nonzero()[0].shape[0] / np.dot(*sparse_rating_matrix.shape)
logging.debug('rating matrix is {:.4%} sparse'.format(sparsity))

for baseline_name, baseline_val in [('zero', np.zeros(sparse_rating_matrix.data.shape[0])), ('mean', np.full(sparse_rating_matrix.data.shape[0], sparse_rating_matrix.data.mean())), ('random', np.random.uniform(low=min(rating_scores), high=max(rating_scores), size=sparse_rating_matrix.data.shape[0]))]:
    log_line = '{:<8s} ::'.format(baseline_name)
    for acc_name, acc in sorted(accuracy_methods.items(), key=lambda x: x[0]):  # Predictable algorithm order for pretty printing
        overall_acc = acc(baseline_val, sparse_rating_matrix)
        log_line += ' | Overall-{0:s}: {overall_acc:>7.2f}'.format(acc_name, overall_acc=overall_acc)

    logging.debug(log_line)

# Keep track of the columns added to the dataframe
algorithms_name = set()

sci_algorithms = {}
sci_algorithms['SciPy-SVD'] = recsys.SciPySVD(**algorithms_args['SciPy-SVD'])
sci_algorithms['SKLearn-SVD'] = recsys.SKLearnSVD(**algorithms_args['SKLearn-SVD'])
sci_algorithms['SKLearn-KNN'] = recsys.SKLearnKNN(**algorithms_args['SKLearn-KNN'])
sci_algorithms['SKLearn-NMF'] = recsys.SKLearnNMF(**algorithms_args['SKLearn-NMF'])
algorithms_name.update(sci_algorithms.keys())
# Initialize a dictionary with an entry for each algorithm which shall store accuracy values for every selected accuracy method
algorithms_error = {}
for alg_name in sci_algorithms.keys():
    # Tuple of training error and test error for each algorithm
    algorithms_error[alg_name] = {acc_name: np.array([0., 0.]) for acc_name in accuracy_methods.keys()}

kf = KFold(n_splits=n_folds, shuffle=True)

i = 0
# The ordering the indices of the matrix and the user_ids, item_ids of the frame must match in order to merge the prediction back into the table
# By default the created sparse matrix has sorted indices. However, act with caution when working with subsets of the matrix!
for train_idx, test_idx in kf.split(sparse_rating_matrix):
    i += 1

    for alg_name, alg in sorted(sci_algorithms.items(), key=lambda x: x[0]):  # Predictable algorithm order for reproducibility
        log_line = '{:<15s} (fold {:>d}/{:<d}) ::'.format(alg_name, i, n_folds)
        train_predictions = alg.fit_transform(sparse_rating_matrix[train_idx].sorted_indices())
        test_predictions = alg.estimate(sparse_rating_matrix[test_idx].sorted_indices())

        # Extract the test predictions from the matrix by selecting the proper indices from the test_idx and the matrix
        user_merge_idx, item_merge_idx = sparse_rating_matrix.nonzero()
        merge_users = np.isin(user_merge_idx, test_idx)  # The test_idx is a subset of the rows of the matrix
        user_merge_idx, item_merge_idx = user_merge_idx[merge_users], item_merge_idx[merge_users]
        items_prediction = pd.DataFrame({'Prediction' + alg_name: np.asarray(test_predictions[sparse_rating_matrix[test_idx].sorted_indices().nonzero()]).flatten(), 'DonorID': user_ids[user_merge_idx], 'ProjectID': item_ids[item_merge_idx]})
        items = pd.merge(items, items_prediction, on=['DonorID', 'ProjectID'], how='left', suffixes=('_x', '_y'), sort=False)
        # Add predictions (NaN is treated as zero here) and remove separate results if necessary
        if 'Prediction' + alg_name + '_x' in items.columns and 'Prediction' + alg_name + '_y' in items.columns:
            items['Prediction' + alg_name] = items[['Prediction' + alg_name + '_x', 'Prediction' + alg_name + '_y']].sum(axis=1)
            items = items.drop(['Prediction' + alg_name + '_x', 'Prediction' + alg_name + '_y'], axis=1)

        for acc_name, acc in sorted(accuracy_methods.items(), key=lambda x: x[0]):  # Predictable algorithm order for pretty printing
            train_acc, test_acc = acc(train_predictions, sparse_rating_matrix[train_idx].sorted_indices()), acc(test_predictions, sparse_rating_matrix[test_idx].sorted_indices())
            algorithms_error[alg_name][acc_name] += np.array([train_acc, test_acc]) / n_folds
            log_line += ' | Training-{0:s}: {train_acc:>7.2f}, Test-{0:s}: {test_acc:>7.2f}'.format(acc_name, train_acc=train_acc, test_acc=test_acc)

        logging.debug(log_line)

for alg_name, acc_methods in sorted(algorithms_error.items(), key=lambda x: x[0]):
    log_line = '{:<15s} (average) ::'.format(alg_name)
    for acc_name, acc_value in sorted(acc_methods.items(), key=lambda x: x[0]):
        log_line += ' | Training-{0:s}: {train_acc:>7.2f}, Test-{0:s}: {test_acc:>7.2f}'.format(acc_name, train_acc=acc_value[0], test_acc=acc_value[1])

    logging.debug(log_line)

spl_algorithms = {}
spl_algorithms['SPL-SVD'] = spl.SVD(**algorithms_args['SPL-SVD'])
spl_algorithms['SPL-SVDpp'] = spl.SVDpp(**algorithms_args['SPL-SVDpp'])
spl_algorithms['SPL-NMF'] = spl.NMF(**algorithms_args['SPL-NMF'])
spl_algorithms['SPL-KNNWithMeans'] = spl.KNNWithMeans(**algorithms_args['SPL-KNNWithMeans'])
spl_algorithms['SPL-KNNBasic'] = spl.KNNBasic(**algorithms_args['SPL-KNNBasic'])
spl_algorithms['SPL-KNNWithZScore'] = spl.KNNWithZScore(**algorithms_args['SPL-KNNWithZScore'])
spl_algorithms['SPL-KNNBaseline'] = spl.KNNBaseline(**algorithms_args['SPL-KNNBaseline'])
spl_algorithms['SPL-NormalPredictor'] = spl.NormalPredictor(**algorithms_args['SPL-NormalPredictor'])
spl_algorithms['SPL-CoClustering'] = spl.CoClustering(**algorithms_args['SPL-CoClustering'])
spl_algorithms['SPL-SlopeOne'] = spl.SlopeOne(**algorithms_args['SPL-SlopeOne'])
algorithms_name.update(spl_algorithms.keys())

# Read data into scikit-surprise respectively surpriselib
spl_reader = spl_Reader(line_format='user item rating', rating_scale=(int(min(rating_scores)), int(max(rating_scores))))
spl_items = spl.Dataset.load_from_df(items[['DonorID', 'ProjectID', 'DonationAmount']], spl_reader)

spl_kf = spl_KFold(n_splits=n_folds, shuffle=True)
for spl_train, spl_test in spl_kf.split(spl_items):
    for alg_name, alg in sorted(spl_algorithms.items(), key=lambda x: x[0]):
        alg.fit(spl_train)
        # Test returns an object of type surprise.prediction_algorithms.predictions.Prediction
        predictions = pd.DataFrame(alg.test(spl_test), columns=['DonorID', 'ProjectID', 'TrueRating', 'Prediction' + alg_name, 'RatingDetails'])
        # Merge the predicted rating into the dataframe
        items = pd.merge(items, predictions[['DonorID', 'ProjectID', 'Prediction' + alg_name]], on=['DonorID', 'ProjectID'], how='left', suffixes=('_x', '_y'), sort=False)
        if 'Prediction' + alg_name + '_x' in items.columns and 'Prediction' + alg_name + '_y' in items.columns:
            items['Prediction' + alg_name] = items[['Prediction' + alg_name + '_x', 'Prediction' + alg_name + '_y']].sum(axis=1)
            items = items.drop(['Prediction' + alg_name + '_x', 'Prediction' + alg_name + '_y'], axis=1)

# Overall accuracy
for alg_name in sorted(algorithms_name):
    log_line = '{:<20s} (overall) ::'.format(alg_name)
    for acc_name, acc in sorted(accuracy_methods.items(), key=lambda x: x[0]):
            overall_acc = acc(items['Prediction' + alg_name].values, np.asarray(items['DonationAmount']))
            log_line += ' | Overall-{0:s}: {overall_acc:>7.2f}'.format(acc_name, overall_acc=overall_acc)

    logging.info(log_line)
