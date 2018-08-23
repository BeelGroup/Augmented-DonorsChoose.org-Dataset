#!/usr/bin/env python3

import logging
import os

import numpy as np
import pandas as pd
import surprise as spl
import yaml

import recsys

with open('config.yml', 'r') as stream:
    config = yaml.load(stream)

# Read in all the configuration once instead of possibly failing later because of missing values
log_level = config['log_level']
donations_filepath = config['donations_filepath']
projects_filepath = config['projects_filepath']
random_state_seed = config['random_state_seed']
n_random_non_interacted_items =  config['n_random_non_interacted_items']
sampling_methods = config['sampling_methods']
n_folds = config['n_folds']
algorithms_args = config['algorithms_args']
accuracy_methods = config['accuracy_methods']
rating_scores = config['rating_scores']
rating_range_quantile = config['rating_range_quantile']

logging.basicConfig(level=log_level)
# Implicitly rely on other commands using the current random state from numpy
np.random.seed(random_state_seed)

donations = pd.read_csv(donations_filepath)
projects = pd.read_csv(projects_filepath)
# Get rid of pesky whitespaces in column names (pandas' query convenience function e.g. is allergic to them)
donations.columns = donations.columns.str.replace(' ', '')
projects.columns = projects.columns.str.replace(' ', '')
# Unconditionally drop donations which have no associated project (why do those kind or transations even exist?!)
donations = donations[donations['ProjectID'].isin(projects['ProjectID'])]

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

collab_filters = recsys.CollaborativeFilters(items, ('DonorID', 'ProjectID', 'DonationAmount'), rating_scores=rating_scores, algorithms_args=algorithms_args, accuracy_methods=accuracy_methods, log_level=log_level)
items = collab_filters.fit_all(n_folds=n_folds, n_random_non_interacted_items=n_random_non_interacted_items).items
algorithms_name.update(collab_filters.algorithms_name)

content_filters = recsys.ContentFilers(items, ('DonorID', 'ProjectID', 'DonationAmount'), projects, ('ProjectTitle', 'ProjectShortDescription', 'ProjectNeedStatement', 'ProjectEssay'), algorithms_args=algorithms_args, accuracy_methods=accuracy_methods, log_level=log_level)
items = content_filters.fit_all(n_random_non_interacted_items=n_random_non_interacted_items).items
algorithms_name.update(content_filters.algorithms_name)
