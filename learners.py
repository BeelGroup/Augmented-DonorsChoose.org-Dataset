#!/usr/bin/env python3

import logging
import os

import numpy as np
import pandas as pd
from surprise import (Dataset, NormalPredictor, BaselineOnly, KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline, SVD, SVDpp, NMF, SlopeOne, CoClustering)
from surprise.model_selection import cross_validate
from surprise.reader import Reader


# Constant variables which might be worth reading in from a configuration file
logging.basicConfig(level=logging.INFO)
items_filepath = os.path.join('data', 'donorschoose.org', 'Donations.csv')
projects_filepath = os.path.join('data', 'donorschoose.org', 'Projects.csv')
n_folds = 5
n_samples = int(1e4)
n_jobs = 2
random_state_seed = 2718281828


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

# Read data into scikit-surprise respectively surpriselib
spl_reader = Reader(line_format='user item rating', rating_scale=(0, 1))
spl_items = Dataset.load_from_df(items[['DonorID', 'SchoolID', 'DonationAmount']], spl_reader)

# Evaluate models
cross_validate(SVD(), spl_items, measures=['RMSE', 'MAE'], cv=n_folds, verbose=True, n_jobs=n_jobs)
cross_validate(SVDpp(), spl_items, measures=['RMSE', 'MAE'], cv=n_folds, verbose=True, n_jobs=n_jobs)
cross_validate(NMF(), spl_items, measures=['RMSE', 'MAE'], cv=n_folds, verbose=True, n_jobs=n_jobs)
cross_validate(KNNWithMeans(), spl_items, measures=['RMSE', 'MAE'], cv=n_folds, verbose=True, n_jobs=n_jobs)
cross_validate(KNNBasic(), spl_items, measures=['RMSE', 'MAE'], cv=n_folds, verbose=True, n_jobs=n_jobs)
cross_validate(KNNWithZScore(), spl_items, measures=['RMSE', 'MAE'], cv=n_folds, verbose=True, n_jobs=n_jobs)
cross_validate(KNNBaseline(), spl_items, measures=['RMSE', 'MAE'], cv=n_folds, verbose=True, n_jobs=n_jobs)
cross_validate(NormalPredictor(), spl_items, measures=['RMSE', 'MAE'], cv=n_folds, verbose=True, n_jobs=n_jobs)
cross_validate(CoClustering(), spl_items, measures=['RMSE', 'MAE'], cv=n_folds, verbose=True, n_jobs=n_jobs)
cross_validate(SlopeOne(), spl_items, measures=['RMSE', 'MAE'], cv=n_folds, verbose=True, n_jobs=n_jobs)
