#!/usr/bin/env python3

import itertools
import logging

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeClassifier

with open('config-meta-learners.yml', 'r') as stream:
    config = yaml.load(stream)

# Read in all the configuration once instead of possibly failing later because of missing values
log_level = config['log_level']
meta_items_filepath = config['meta_items_filepath']
donations_filepath = config['donations_filepath']
projects_filepath = config['projects_filepath']
donors_filepath = config['donors_filepath']
output_filepath = config['output_filepath']
random_state_seed = config['random_state_seed']
algorithms_name = config['algorithms_name']
algorithms_accuracy_name = config['algorithms_accuracy_name']
n_splits = config['n_splits']
train_size = config['train_size']

logging.basicConfig(level=log_level)
# Implicitly rely on other commands using the current random state from numpy
np.random.seed(random_state_seed)

meta_items = pd.read_csv(meta_items_filepath)
donations = pd.read_csv(donations_filepath)
projects = pd.read_csv(projects_filepath)
donors = pd.read_csv(donors_filepath)
# Get rid of pesky whitespaces in column names (pandas' query convenience function e.g. is allergic to them)
meta_items.columns = meta_items.columns.str.replace(' ', '')
donations.columns = donations.columns.str.replace(' ', '')
projects.columns = projects.columns.str.replace(' ', '')
donors.columns = donors.columns.str.replace(' ', '')

# Unconditionally drop transactions which have no associated item (why do those kind or entries even exist?!)
donations = donations[donations['ProjectID'].isin(projects['ProjectID'])]
# Drop duplicate transactions
donations = donations.drop_duplicates(subset=['DonorID', 'ProjectID'], keep='first')
donations['DonationReceivedDate'] = pd.to_datetime(donations['DonationReceivedDate'])

# Keep track of all columns which shall be used for training
feature_columns = set()

# Preprocessing: Add additional information about the item, user and transaction to the meta-features table
projects_cat_columns = ['ProjectSubjectCategoryTree', 'ProjectSubjectSubcategoryTree', 'ProjectResourceCategory', 'ProjectGradeLevelCategory']
donors_cat_columns = ['DonorState', 'DonorCity', 'DonorZip', 'DonorIsTeacher']
for df_cat, cat_columns, merge_on_column in [
    (projects, projects_cat_columns, 'ProjectID'),
    (donors, donors_cat_columns, 'DonorID')
]:
    for c in cat_columns:
        df_cat[c] = df_cat[c].astype('category').cat.codes

    meta_items = pd.merge(meta_items, df_cat[np.append(cat_columns, merge_on_column)], on=merge_on_column, how='left', sort=False)

feature_columns.update(projects_cat_columns, donors_cat_columns)

# Preprocessing: Merge in the day of week in addition to the hour and minute as further meta-feature
donations['DonationReceivedDateDayOfWeek'] = donations['DonationReceivedDate'].dt.dayofweek
donations['DonationReceivedDateTimeOfDay'] = donations['DonationReceivedDate'].dt.hour * 60 + donations['DonationReceivedDate'].dt.minute
meta_items = pd.merge(meta_items, donations[['DonorID', 'ProjectID', 'DonationReceivedDateDayOfWeek', 'DonationReceivedDateTimeOfDay']], on=['DonorID', 'ProjectID'], how='left', sort=False)
feature_columns.update(['DonationReceivedDateDayOfWeek', 'DonationReceivedDateTimeOfDay'])

i = 0
rs = ShuffleSplit(n_splits=n_splits, train_size=train_size, test_size=None)
for train_idx, test_idx in rs.split(meta_items):
    i += 1

    # Preprocessing: Add further information to the meta-features table which is test-train specific
    val_count_columns = ['DonorID', 'ProjectID']
    for idx, c in itertools.product([train_idx, test_idx], val_count_columns):
        value_counts = meta_items.loc[idx][c].value_counts(sort=False).reset_index().rename(columns={'index': c, c: 'ValueCounts' + c})
        meta_items.at[idx, 'ValueCounts' + c] = pd.merge(meta_items[[c]], value_counts, on=c, how='left', sort=False).loc[idx]['ValueCounts' + c]

    for acc_name in algorithms_accuracy_name:
        # Convert between column names
        alg_acc_to_alg_columns = {acc_name + alg_name: alg_name for alg_name in algorithms_name}

        alg_comparison = meta_items.loc[test_idx][sorted(alg_acc_to_alg_columns.keys())].mean(axis=0)
        best_alg_idx = alg_comparison.idxmin()
        best_alg = alg_acc_to_alg_columns[best_alg_idx], alg_comparison[best_alg_idx]
        combined_best = 'combined', meta_items.loc[test_idx][sorted(alg_acc_to_alg_columns.keys())].min(axis=1).mean()

        logging.info('{acc_name:<16s} ({:^15s}) (shuffle {i:>d}/{n_splits:<d}) :: | {:^25s} | Test-{acc_name}: {:>7.2f}'.format('overall best', *best_alg, i=i, n_splits=n_splits, acc_name=acc_name))
        logging.info('{acc_name:<16s} ({:^15s}) (shuffle {i:>d}/{n_splits:<d}) :: | {:^25s} | Test-{acc_name}: {:>7.2f}'.format('combined best', *combined_best, i=i, n_splits=n_splits, acc_name=acc_name))

        meta_alg_name, meta_alg = 'DecisionTree', DecisionTreeClassifier()
        for alg_name in algorithms_name:
            meta_alg.fit(meta_items.loc[train_idx][sorted(feature_columns)], meta_items.loc[train_idx][acc_name + alg_name])
            meta_items.at[test_idx, 'Prediction' + meta_alg_name + acc_name + alg_name] = meta_alg.predict(meta_items.loc[test_idx][sorted(feature_columns)])

            err = meta_items.loc[test_idx]['Prediction' + meta_alg_name + acc_name + alg_name] - meta_items.loc[test_idx][acc_name + alg_name]
            rmse = np.sqrt(np.square(err).mean())
            mae = np.abs(err).mean()
            logging.debug('Prediction {acc_name:<16s}{alg_name:<25s} (shuffle {i:>d}/{n_splits:<d}) :: | Test-RMSE: {:>7.2f} | Test-MAE: {:>7.2f}'.format(rmse, mae, i=i, n_splits=n_splits, acc_name=acc_name, alg_name=alg_name))

        meta_alg_to_alg_acc_columns = {'Prediction' + meta_alg_name + acc_name + alg_name: acc_name + alg_name for alg_name in algorithms_name}
        # Look up the lowest prediction made by the meta-learning algorithm and select the column which contains the actual result
        predict_alg_acc_columns = meta_items.loc[test_idx][sorted(meta_alg_to_alg_acc_columns.keys())].idxmin(axis=1).astype('category').cat.rename_categories(meta_alg_to_alg_acc_columns)

        meta_items.at[test_idx, 'MetaPrediction' + meta_alg_name] = meta_items.lookup(test_idx, predict_alg_acc_columns)
        best_meta = 'meta', meta_items.loc[test_idx]['MetaPrediction' + meta_alg_name].mean()
        logging.info('{acc_name:<16s} ({:^15s}) (shuffle {i:>d}/{n_splits:<d}) :: | {:^25s} | Test-{acc_name}: {:>7.2f}'.format('meta', *best_meta, i=i, n_splits=n_splits, acc_name=acc_name))

if output_filepath is not None and output_filepath is not False:
    meta_items.to_csv(output_filepath)
