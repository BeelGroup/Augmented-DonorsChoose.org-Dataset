#!/usr/bin/env python3

import logging

import numpy as np
import pandas as pd
import yaml

import recsys

with open('config.yml', 'r') as stream:
    config = yaml.load(stream)

# Read in all the configuration once instead of possibly failing later because of missing values
log_level = config['log_level']
donations_filepath = config['donations_filepath']
projects_filepath = config['projects_filepath']
donors_filepath = config['donors_filepath']
output_filepath = config['output_filepath']
random_state_seed = config['random_state_seed']
n_random_non_interacted_items = config['n_random_non_interacted_items']
sampling_methods = config['sampling_methods']
n_folds = config['n_folds']
algorithms_args = config['algorithms_args']
groupby = config['groupby']
accuracy_methods = config['accuracy_methods']
rating_scores = config['rating_scores']
rating_range_quantile = config['rating_range_quantile']

logging.basicConfig(level=log_level)
# Implicitly rely on other commands using the current random state from numpy
np.random.seed(random_state_seed)

donations = pd.read_csv(donations_filepath)
projects = pd.read_csv(projects_filepath)
donors = pd.read_csv(donors_filepath)
# Get rid of pesky whitespaces in column names (pandas' query convenience function e.g. is allergic to them)
for df in [donations, projects, donors]:
    df.columns = df.columns.str.replace(' ', '')

# Preserve a subset of the table which accumulates duplicate donations to merge them back into the cleansed table
# Hence, do not try to predict projects which the donor has already donated to
donations_donation_amount = donations.groupby(['DonorID', 'ProjectID'])['DonationAmount'].sum().reset_index()

# Unconditionally drop transactions which have no associated item (why do those kind or entries even exist?!)
donations = donations[donations['ProjectID'].isin(projects['ProjectID'])]
# Drop duplicate transactions
donations = donations.drop_duplicates(subset=['DonorID', 'ProjectID'], keep='first')
donations['DonationReceivedDate'] = pd.to_datetime(donations['DonationReceivedDate'])
# Remove special delimiter keyword from item description
projects['ProjectEssay'] = projects['ProjectEssay'].str.replace('<!--DONOTREMOVEESSAYDIVIDER-->', '', regex=False)

# Update the transaction strength to have the value of the sum of the transactions
items = pd.merge(donations[donations.columns.difference(['DonationAmount'])], donations_donation_amount, on=['DonorID', 'ProjectID'], how='left', sort=False)

# Add information about the user
donors_columns = ['DonorState', 'DonorCity', 'DonorZip', 'DonorIsTeacher']
items = pd.merge(items, donors[np.append('DonorID', donors_columns)], on='DonorID', how='left', sort=False)

# Apply the cleaning and sampling operations in a fixed order independently of the order of the dict or the user's choice
sampling_methods_priority = {'drop_raw_values': 200, 'frequency_boundaries': 500, 'sample': 900}
for method, opt in sorted(sampling_methods.items(), key=lambda x: sampling_methods_priority[x[0]]):
    if opt is None or opt is False:
        pass
    elif method == 'drop_raw_values':
        for drop_query in opt:
            items = items.drop(items.query(drop_query).index)
    elif method == 'frequency_boundaries':
        # Note, this does not ensure that several conditions are met at the same time; Just selecting the intersection is not a solution!
        for column, bound in opt:
            value_counts = items[column].value_counts()
            items = items[items[column].isin(value_counts.index[value_counts >= bound])]
    elif method == 'sample':
        # In case a frequency boundary is selected, ensure that the last condition is met in the sampled data; Thereby sample at least as many as specified but possibly more items
        if 'frequency_boundaries' in sampling_methods.keys() and len(sampling_methods['frequency_boundaries']) > 0:
            column, bound = sampling_methods['frequency_boundaries'][-1]
            value_counts = items[column].value_counts(sort=False).sample(frac=1.)

            selection_limit = 1
            while value_counts.iloc[:selection_limit].sum() < opt:
                selection_limit += 1

            items = items[items[column].isin(value_counts.iloc[:selection_limit].index)]
        else:
            items = items.sample(n=opt)
    else:
        raise ValueError('Expected a valid sampling method from ' + str(sampling_methods_priority.keys()) + ', got "' + str(method) + '"')

logging.debug('{:d} unique donors donated to {:d} unique projects'.format(items['DonorID'].unique().shape[0], items['ProjectID'].unique().shape[0]))
# Convert DonationAmount into a rating
rating_bins = np.logspace(*np.log10(items['DonationAmount'].quantile(rating_range_quantile).values), num=len(rating_scores) + 1)
rating_bins[0], rating_bins[-1] = items['DonationAmount'].min(), items['DonationAmount'].max()
items['DonationAmount'] = pd.cut(items['DonationAmount'], bins=rating_bins, include_lowest=True, labels=rating_scores, retbins=False).astype(np.float)

algorithms_name = set()

for group in [{'columns': 'DonorID', 'algorithms_args': algorithms_args}] + groupby:
    if type(group['columns']) is str:
        item_columns = (group['columns'], 'ProjectID', 'DonationAmount')
    elif type(group['columns']) is list:
        item_columns = ('Concat' + ''.join(group['columns']), 'ProjectID', 'DonationAmount')
        items[item_columns[0]] = ''
        for c in group['columns']:
            items[item_columns[0]] += items[c].astype(str)
    else:
        raise ValueError('Got an unexpected type of `columns` to groupby, expected either `str` or `list`, got "' + str(type(group['columns'])) + '"')

    collab_filters = recsys.CollaborativeFilters(items, item_columns, rating_scores=rating_scores, algorithms_args=group['algorithms_args'], accuracy_methods=accuracy_methods, log_level=log_level)
    items = collab_filters.fit_all(n_folds=n_folds, n_random_non_interacted_items=n_random_non_interacted_items).items
    algorithms_name.update(collab_filters.algorithms_name)

    content_filters = recsys.ContentFilers(items, item_columns, projects, ('ProjectTitle', 'ProjectShortDescription', 'ProjectNeedStatement', 'ProjectEssay'), algorithms_args=group['algorithms_args'], accuracy_methods=accuracy_methods, log_level=log_level)
    items = content_filters.fit_all(n_random_non_interacted_items=n_random_non_interacted_items).items
    algorithms_name.update(content_filters.algorithms_name)

if output_filepath is not None and output_filepath is not False:
    items.to_csv(output_filepath)
