#!/usr/bin/env python3

import itertools
import logging

import numpy as np
import pandas as pd
from sklearn import tree, ensemble
import yaml
from keras import optimizers
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import ShuffleSplit


class NNHelper(object):
    def __init__(self, **options):
        self.options = {'epochs': 50, 'batch_size': 256, 'optimizer': 'adadelta', 'loss': 'categorical_crossentropy', 'metrics': ['accuracy'], 'verbose': 1}
        self.options.update(options)

        self.model = None
        self.categories = None

    def fit(self, X, y):
        # Convert categories to hot values
        self.categories = None
        if y.dtype.name == 'category':
            self.categories = y.cat.categories
            y = y.cat.codes

        y_hot = to_categorical(y)

        self.model = Sequential()
        self.model.add(Dense(units=X.shape[1] * 4, activation='relu', input_dim=X.shape[1]))
        self.model.add(Dense(units=y_hot.shape[1], activation='softmax'))
        self.model.compile(optimizer=self.options['optimizer'], loss=self.options['loss'], metrics=self.options['metrics'])

        self.model.fit(X, y_hot, epochs=self.options['epochs'], batch_size=self.options['batch_size'], verbose=self.options['verbose'])

    def predict(self, X):
        y_hot = self.model.predict(X)

        y = np.argmax(y_hot, axis=1)
        # Convert back to category names if necessary
        if self.categories is not None:
            y = self.categories[y]

        return  y


with open('config-meta-learners.yml', 'r') as stream:
    config = yaml.load(stream)

# Read in all the configuration once instead of possibly failing later because of missing values
log_level = config['log_level']
meta_items_filepath = config['meta_items_filepath']
donations_filepath = config['donations_filepath']
projects_filepath = config['projects_filepath']
donors_filepath = config['donors_filepath']
schools_filepath = config['schools_filepath']
output_filepath = config['output_filepath']
random_state_seed = config['random_state_seed']
algorithms_name = config['algorithms_name']
algorithms_accuracy_name = config['algorithms_accuracy_name']
algorithm_selection_methods = config['algorithm_selection_methods']
n_splits = config['n_splits']
train_size = config['train_size']
meta_algorithms_args = config['meta_algorithms_args']

logging.basicConfig(level=log_level)
# Implicitly rely on other commands using the current random state from numpy
np.random.seed(random_state_seed)

meta_items = pd.read_csv(meta_items_filepath)
donations = pd.read_csv(donations_filepath)
projects = pd.read_csv(projects_filepath)
donors = pd.read_csv(donors_filepath)
schools = pd.read_csv(schools_filepath)
# Get rid of pesky whitespaces in column names (pandas' query convenience function e.g. is allergic to them)
meta_items.columns = meta_items.columns.str.replace(' ', '')
donations.columns = donations.columns.str.replace(' ', '')
projects.columns = projects.columns.str.replace(' ', '')
donors.columns = donors.columns.str.replace(' ', '')
schools.columns = schools.columns.str.replace(' ', '')

# Unconditionally drop transactions which have no associated item (why do those kind or entries even exist?!)
donations = donations[donations['ProjectID'].isin(projects['ProjectID'])]
# Drop duplicate transactions
donations = donations.drop_duplicates(subset=['DonorID', 'ProjectID'], keep='first')
donations['DonationReceivedDate'] = pd.to_datetime(donations['DonationReceivedDate'])

# Keep track of all columns which shall be used for training
feature_columns = set()

# Allow merges via the 'SchoolID' column by adding the required information to the meta-features table
meta_items = pd.merge(meta_items, projects[['ProjectID', 'SchoolID']], on='ProjectID', how='left', sort=False)
# Preprocessing: Add additional information about the item, user and transaction to the meta-features table
projects_columns = ['ProjectSubjectCategoryTree', 'ProjectSubjectSubcategoryTree', 'ProjectResourceCategory', 'ProjectGradeLevelCategory']
donors_columns = ['DonorState', 'DonorCity', 'DonorZip', 'DonorIsTeacher']
schools_columns = ['SchoolMetroType', 'SchoolPercentageFreeLunch', 'SchoolState', 'SchoolCity', 'SchoolZip']
for df_cat, columns, merge_on_column in [(projects, projects_columns, 'ProjectID'), (donors, donors_columns, 'DonorID'), (schools, schools_columns, 'SchoolID')]:
    meta_items = pd.merge(meta_items, df_cat[np.append(columns, merge_on_column)], on=merge_on_column, how='left', sort=False)

# Convert categorical text columns to integer values
cat_columns = ['ProjectSubjectCategoryTree', 'ProjectSubjectSubcategoryTree', 'ProjectResourceCategory', 'ProjectGradeLevelCategory', 'SchoolMetroType']
for c in cat_columns:
    meta_items[c] = meta_items[c].astype('category').cat.codes

# Translate shared column values to common integers
shared_columns = [['SchoolState', 'DonorState'], ['SchoolCity', 'DonorCity']]
for column_tuple in shared_columns:
    cat = set()
    for c in column_tuple:
        cat.update(meta_items[c].astype('category').cat.categories)

    # Sort set in order to get predictable results and to make the distance somewhat meaningful
    shared_map = dict(zip(sorted(cat), np.arange(len(cat))))
    meta_items[column_tuple] = meta_items[column_tuple].replace(shared_map)

# Cut the zip code of the school to be of the same shape as for the user
meta_items['SchoolZip'] = meta_items['SchoolZip'].astype(str).str[0:3].astype(int)
# Convert teacher status to 1 / 0
meta_items['DonorIsTeacher'] = meta_items['DonorIsTeacher'].map({'Yes': 1, 'No': 0})

feature_columns.update(projects_columns, donors_columns, schools_columns)

# Preprocessing: Merge in the information about the date as further meta-features
donations['DonationReceivedDateYear'] = donations['DonationReceivedDate'].dt.year
donations['DonationReceivedDateMonth'] = donations['DonationReceivedDate'].dt.month
donations['DonationReceivedDateDay'] = donations['DonationReceivedDate'].dt.day
donations['DonationReceivedDateDayOfWeek'] = donations['DonationReceivedDate'].dt.dayofweek
donations['DonationReceivedDateTimeOfDay'] = donations['DonationReceivedDate'].dt.hour * 60 + donations['DonationReceivedDate'].dt.minute
date_columns = ['DonationReceivedDateYear','DonationReceivedDateMonth', 'DonationReceivedDateDay', 'DonationReceivedDateDayOfWeek', 'DonationReceivedDateTimeOfDay']
meta_items = pd.merge(meta_items, donations[['DonorID', 'ProjectID', *date_columns]], on=['DonorID', 'ProjectID'], how='left', sort=False)
feature_columns.update(date_columns)

# Fill remaining NaN values (in DonorCity, DonorZip, SchoolCity and SchoolPercentageFreeLunch) with a value otherwise not used: -1
# Most algorithms will not care about NaN either way but some are allergic to it
meta_items[sorted(feature_columns)] = meta_items[sorted(feature_columns)].fillna(-1.)

meta_algorithms = {}
meta_algorithms_avail = {'SKLearn-AdaBoostClassifier': ensemble.AdaBoostClassifier, 'SKLearn-BaggingClassifier': ensemble.BaggingClassifier,
    'SKLearn-ExtraTreesClassifier': ensemble.ExtraTreesClassifier, 'SKLearn-GradientBoostingClassifier': ensemble.GradientBoostingClassifier,
    'SKLearn-RandomForestClassifier': ensemble.RandomForestClassifier, 'SKLearn-DecisionTreeClassifier': tree.DecisionTreeClassifier,
    'SKLearn-ExtraTreeClassifier': tree.ExtraTreeClassifier, 'Keras-NN': NNHelper}

# Initialize all selected meta-algorithms
for meta_alg_name in np.intersect1d(list(meta_algorithms_avail.keys()), list(meta_algorithms_args.keys())):
    meta_algorithms[meta_alg_name] = meta_algorithms_avail[meta_alg_name](**meta_algorithms_args[meta_alg_name])

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

        for meta_alg_name, meta_alg in sorted(meta_algorithms.items(), key=lambda x: x[0]):
            if 'accuracy prediction' in algorithm_selection_methods:
                for alg_name in algorithms_name:
                    meta_alg.fit(meta_items.loc[train_idx][sorted(feature_columns)], meta_items.loc[train_idx][acc_name + alg_name])
                    meta_items.at[test_idx, 'Prediction' + meta_alg_name + acc_name + alg_name] = meta_alg.predict(meta_items.loc[test_idx][sorted(feature_columns)])

                    err = meta_items.loc[test_idx]['Prediction' + meta_alg_name + acc_name + alg_name] - meta_items.loc[test_idx][acc_name + alg_name]
                    rmse = np.sqrt(np.square(err).mean())
                    mae = np.abs(err).mean()
                    logging.debug('{meta_alg_name:<35s} ({:^15s}) (shuffle {i:>d}/{n_splits:<d}) {acc_name:<16s}{alg_name:<25s} :: | Test-RMSE: {:>7.2f} | Test-MAE: {:>7.2f}'.format('meta acc', rmse, mae, i=i, n_splits=n_splits, acc_name=acc_name, alg_name=alg_name, meta_alg_name=meta_alg_name))

                meta_alg_to_alg_acc_columns = {'Prediction' + meta_alg_name + acc_name + alg_name: acc_name + alg_name for alg_name in algorithms_name}
                # Look up the lowest prediction made by the meta-learning algorithm and select the column which contains the actual result
                predict_alg_acc_columns = meta_items.loc[test_idx][sorted(meta_alg_to_alg_acc_columns.keys())].idxmin(axis=1).astype('category').cat.rename_categories(meta_alg_to_alg_acc_columns)

                meta_items.at[test_idx, 'MetaPrediction' + meta_alg_name] = meta_items.lookup(test_idx, predict_alg_acc_columns)
                best_meta = 'meta accuracy', meta_items.loc[test_idx]['MetaPrediction' + meta_alg_name].mean()
                logging.info('{meta_alg_name:<35s} ({:^15s}) (shuffle {i:>d}/{n_splits:<d}) :: | {:^25s} | Test-{acc_name}: {:>7.2f}'.format('meta acc', *best_meta, i=i, n_splits=n_splits, acc_name=acc_name, meta_alg_name=meta_alg_name))

            if 'classification' in algorithm_selection_methods:
                # Treat the best performing algorithm of a row as the row's class
                meta_items['SubalgorithmCategory'] = meta_items[sorted(alg_acc_to_alg_columns.keys())].idxmin(axis=1).astype('category')

                # Pass the raw category name to the fit method as it is expected that it can cope with strings
                meta_alg.fit(meta_items.loc[train_idx][sorted(feature_columns)], meta_items.loc[train_idx]['SubalgorithmCategory'])
                meta_items.at[test_idx, 'SubalgorithmPrediction' + meta_alg_name + acc_name] = meta_alg.predict(meta_items.loc[test_idx][sorted(feature_columns)])

                classification_accuracy = (meta_items.loc[test_idx]['SubalgorithmCategory'] == meta_items.loc[test_idx]['SubalgorithmPrediction' + meta_alg_name + acc_name]).mean()
                logging.debug('{meta_alg_name:<35s} ({:^15s}) (shuffle {i:>d}/{n_splits:<d}) {acc_name:<16s} :: | Test-Accuracy: {:>7.2%}'.format('meta class', classification_accuracy, i=i, n_splits=n_splits, acc_name=acc_name, meta_alg_name=meta_alg_name))

                meta_items.at[test_idx, 'MetaSubalgorithmPrediction' + meta_alg_name] = meta_items.lookup(test_idx, meta_items.loc[test_idx]['SubalgorithmPrediction' + meta_alg_name + acc_name])
                best_meta = 'meta classify', meta_items.loc[test_idx]['MetaSubalgorithmPrediction' + meta_alg_name].mean()
                logging.info('{meta_alg_name:<35s} ({:^15s}) (shuffle {i:>d}/{n_splits:<d}) :: | {:^25s} | Test-{acc_name}: {:>7.2f}'.format('meta class', *best_meta, i=i, n_splits=n_splits, acc_name=acc_name, meta_alg_name=meta_alg_name))

if output_filepath is not None and output_filepath is not False:
    meta_items.to_csv(output_filepath)
