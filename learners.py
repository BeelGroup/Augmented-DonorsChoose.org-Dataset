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
n_jobs = 2
n_svd_components = 100
n_knn_neighbors = 40
n_samples = int(1e4)
n_folds = 5

# Implicitly rely on other commands using the current random state from numpy
np.random.seed(random_state_seed)

items = pd.read_csv(items_filepath)
projects = pd.read_csv(projects_filepath)
# Get rid of pesky whitespaces in column names
items.columns = items.columns.str.replace(' ', '')
projects.columns = projects.columns.str.replace(' ', '')
# Sample the data as to reduce the amount of required memory
items = items.sample(n=n_samples)

items = pd.merge(items, projects[['ProjectID', 'SchoolID']], on='ProjectID', how='inner', sort=False)
logging.info('%d unique donors donated to %d unique projects respectively %d unique schools' % (len(items['DonorID'].unique()), len(items['ProjectID'].unique()), len(items['SchoolID'].unique())))
# Convert DonationAmount into a 0/1 rating
items.query('DonationAmount > 0')['DonationAmount'] = 1

# Create a sparse matrix for further analysis
user_ids = items['DonorID'].unique()
item_ids = items['SchoolID'].unique()
ratings = items['DonationAmount'].values
row = items['DonorID'].astype(pd.api.types.CategoricalDtype(categories=user_ids)).cat.codes
col = items['SchoolID'].astype(pd.api.types.CategoricalDtype(categories=item_ids)).cat.codes
# Utilize a Compressed Sparse Row matrix as most users merely donate once or twice
sparse_rating_matrix = csr_matrix((ratings, (row, col)), shape=(user_ids.shape[0], item_ids.shape[0]))

sparsity = 1.0 - sparse_rating_matrix.nonzero()[0].shape[0] / np.dot(*sparse_rating_matrix.shape)
logging.info('rating matrix is %.4f%% sparse' % (sparsity * 100))

kf = KFold(n_splits=n_folds, shuffle=True)

i = 0
svd_scipy = recsys.SciPySVD(n_components=n_svd_components)
svd_sklearn = recsys.SKLearnSVD(n_components=n_svd_components)
knn = recsys.SKLearnKNN(n_neighbors=n_knn_neighbors)
# The indices of the matrix and the user_ids, item_ids must match in order to get the merge the prediction back into the frame
for train_idx, test_idx in kf.split(sparse_rating_matrix):
    i += 1
    # Perform a SVD on the training data
    train_predictions = svd_scipy.fit_transform(sparse_rating_matrix[train_idx])
    test_predictions = svd_scipy.estimate(sparse_rating_matrix[test_idx])

    train_rmse, train_mae = recsys.rmse(train_predictions, sparse_rating_matrix[train_idx]), recsys.mae(train_predictions, sparse_rating_matrix[train_idx])
    test_rmse, test_mae = recsys.rmse(test_predictions, sparse_rating_matrix[test_idx]), recsys.mae(test_predictions, sparse_rating_matrix[test_idx])
    logging.info('SciPy-SVD\t (fold %d/%d) :: Training-RMSE: %.2f, Training-MAE: %.2f, Validation-RMSE: %.2f, Validation-MAE: %.2f' % (i, n_folds, train_rmse, train_mae, test_rmse, test_mae))

    train_predictions = svd_sklearn.fit_transform(sparse_rating_matrix[train_idx])
    test_predictions = svd_sklearn.estimate(sparse_rating_matrix[test_idx])

    train_rmse, train_mae = recsys.rmse(train_predictions, sparse_rating_matrix[train_idx]), recsys.mae(train_predictions, sparse_rating_matrix[train_idx])
    test_rmse, test_mae = recsys.rmse(test_predictions, sparse_rating_matrix[test_idx]), recsys.mae(test_predictions, sparse_rating_matrix[test_idx])
    logging.info('SKLearn-SVD\t (fold %d/%d) :: Training-RMSE: %.2f, Training-MAE: %.2f, Validation-RMSE: %.2f, Validation-MAE: %.2f' % (i, n_folds, train_rmse, train_mae, test_rmse, test_mae))

    train_predictions = knn.fit_transform(sparse_rating_matrix[train_idx])
    test_predictions = knn.estimate(sparse_rating_matrix[test_idx])

    train_rmse, train_mae = recsys.rmse(train_predictions, sparse_rating_matrix[train_idx]), recsys.mae(train_predictions, sparse_rating_matrix[train_idx])
    test_rmse, test_mae = recsys.rmse(test_predictions, sparse_rating_matrix[test_idx]), recsys.mae(test_predictions, sparse_rating_matrix[test_idx])
    logging.info('SKLearn-KNN\t (fold %d/%d) :: Training-RMSE: %.2f, Training-MAE: %.2f, Validation-RMSE: %.2f, Validation-MAE: %.2f' % (i, n_folds, train_rmse, train_mae, test_rmse, test_mae))
