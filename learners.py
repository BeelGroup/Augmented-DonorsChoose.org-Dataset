#!/usr/bin/env python3

import logging
import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold


def rmse(estimate, truth):
	estimate = np.asarray(estimate[truth.nonzero()]).flatten()
	truth = np.asarray(truth[truth.nonzero()]).flatten()
	return np.sqrt(mean_squared_error(estimate, truth))


def mae(estimate, truth):
	estimate = np.asarray(estimate[truth.nonzero()]).flatten()
	truth = np.asarray(truth[truth.nonzero()]).flatten()
	return mean_absolute_error(estimate, truth)


# Constant variables which might be worth reading in from a configuration file
logging.basicConfig(level=logging.INFO)
items_filepath = os.path.join('data', 'donorschoose.org', 'Donations.csv')
projects_filepath = os.path.join('data', 'donorschoose.org', 'Projects.csv')
random_state_seed = 2718281828
n_jobs = 2
n_svd_components = 100
n_samples = int(1e4)
n_folds = 2


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
logging.info('%d unique donors donated to %d unique projects respectively %d unique schools'%(len(items['DonorID'].unique()), len(items['ProjectID'].unique()), len(items['SchoolID'].unique())))
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
logging.info('rating matrix is %.4f%% sparse'%(sparsity * 100))


kf = KFold(n_splits=n_folds, shuffle=True)

# The indices of the matrix and the user_ids, item_ids must match in order to get the merge the prediction back into the frame
for train_idx, test_idx in kf.split(sparse_rating_matrix):
	# Perform a SVD on the training data; Skip calculating u as it is not used afterwards
	u, s, vt = svds(sparse_rating_matrix[train_idx], k=n_svd_components, return_singular_vectors='vh')

	# Matrix 'u' corresponds to the user features and shall now be replaced by the test users
	# Hence, to get our new 'u' from the input, one inverts 'vt' and 's'. Bear in mind, 's' is a real quadratic diagonal matrix while 'vt' is a non-quadratic unitary matrix.
	rating_predictions = sparse_rating_matrix[test_idx].dot(vt.T).dot(vt)

	logging.info('SVD :: RMSE: %.2f, MAE: %.2f'%(rmse(rating_predictions, sparse_rating_matrix[test_idx]), mae(rating_predictions, sparse_rating_matrix[test_idx])))
