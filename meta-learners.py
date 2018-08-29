#!/usr/bin/env python3

import logging

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import ShuffleSplit

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
n_random_non_interacted_items = config['n_random_non_interacted_items']
accuracy_methods = config['accuracy_methods']
rating_scores = config['rating_scores']
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

# Preprocessing: Merge in the day of week in addition to the hour and minute as further meta-feature
donations['DonationReceivedDateDayOfWeek'] = donations['DonationReceivedDate'].dt.dayofweek
donations['DonationReceivedDateTimeOfDay'] = donations['DonationReceivedDate'].dt.hour * 60 + donations['DonationReceivedDate'].dt.minute
meta_items = pd.merge(meta_items, donations[['DonorID', 'ProjectID', 'DonationReceivedDateDayOfWeek', 'DonationReceivedDateTimeOfDay']], on=['DonorID', 'ProjectID'], how='left', sort=False)

rs = ShuffleSplit(n_splits=n_splits, train_size=train_size, test_size=None)
for train_idx, test_idx in rs.split(meta_items):
    # Preprocessing: Add further information to the meta-features table which is test-train specific
    val_count_columns = ['DonorID', 'ProjectID']
    for idx in [train_idx, test_idx]:
        for c in val_count_columns:
            value_counts = meta_items.loc[idx][c].value_counts(sort=False).reset_index().rename(columns={'index': c, c: 'ValueCounts' + c})
            meta_items.at[idx, 'ValueCounts' + c] = pd.merge(meta_items[[c]], value_counts, on=c, how='left', sort=False).loc[idx]['ValueCounts' + c]

if output_filepath is not None and output_filepath is not False:
    meta_items.to_csv(output_filepath)
