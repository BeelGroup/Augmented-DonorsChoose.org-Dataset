#!/usr/bin/env python3

import argparse
import logging

import numpy as np
import pandas as pd
import yaml
from keras import optimizers
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from scipy import stats
from sklearn import cluster, ensemble, tree
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import ShuffleSplit


class NNHelper(object):
    def __init__(self, **options):
        self.options = {'epochs': 50, 'batch_size': 256, 'optimizer': 'adadelta', 'loss': 'categorical_crossentropy', 'metrics': ['accuracy'], 'verbose': 1}
        self.options.update(options)

        self.model = None
        self.categories = None

    def fit(self, X, y, **kwargs):
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

        self.model.fit(X, y_hot, epochs=self.options['epochs'], batch_size=self.options['batch_size'], verbose=self.options['verbose'], **kwargs)

    def predict(self, X, **kwargs):
        y_hot = self.model.predict(X, **kwargs)

        y = np.argmax(y_hot, axis=1)
        # Convert back to category names if necessary
        if self.categories is not None:
            y = self.categories[y]

        return  y


class UserCluster(object):
    def __init__(self, **options):
        class_options = {k: v for k, v in options.items() if k != 'getattr_class_name'}
        self.model = getattr(cluster, options['getattr_class_name'])(**class_options)
        self.labels_to_target = None

    def fit(self, X, y=None, **kwargs):
        cluster_labels = self.model.fit_predict(X)

        labels_to_targets_dict = {}
        for label in np.unique(cluster_labels):
            # Simple majority voting to select the first most frequent target for each cluster
            label_pos = np.where(cluster_labels == label)[0]
            labels_to_targets_dict[label] = stats.mode(y[label_pos])[0][0]

        self.labels_to_targets = np.vectorize(labels_to_targets_dict.get)

    def predict(self, X, **kwargs):
        cluster_labels = self.model.predict(X)

        return self.labels_to_targets(cluster_labels)


parser = argparse.ArgumentParser(description='Configurable meta-learning system for data augmentation.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-i', '--input-config', dest='config_path', action='store', type=str, default='config-meta-learners.yml')
args = parser.parse_args()

with open(args.config_path, 'r') as stream:
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
class_penalty_base = config['class_penalty_base']
algorithms_name = config['algorithms_name']
algorithms_accuracy_name = config['algorithms_accuracy_name']
n_splits = config['n_splits']
train_size = config['train_size']
meta_algorithms_args = config['meta_algorithms_args']

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(log_level)
# Implicitly rely on other commands using the current random state from numpy
np.random.seed(random_state_seed)

meta_items = pd.read_csv(meta_items_filepath)
donations = pd.read_csv(donations_filepath)
projects = pd.read_csv(projects_filepath)
donors = pd.read_csv(donors_filepath)
schools = pd.read_csv(schools_filepath)
# Get rid of pesky whitespaces in column names (pandas' query convenience function e.g. is allergic to them)
for df in [donations, projects, donors, schools]:
    df.columns = df.columns.str.replace(' ', '')

# Unconditionally drop transactions which have no associated item (why do those kind or entries even exist?!)
donations = donations[donations['ProjectID'].isin(projects['ProjectID'])]
# Drop duplicate transactions
donations = donations.drop_duplicates(subset=['DonorID', 'ProjectID'], keep='first')
donations['DonationReceivedDate'] = pd.to_datetime(donations['DonationReceivedDate'])

# Allow merges via the 'SchoolID' column by adding the required information to the meta-features table
if 'SchoolID' not in meta_items.columns:
    meta_items = pd.merge(meta_items, projects[['ProjectID', 'SchoolID']], on='ProjectID', how='left', sort=False)

# Preprocessing: Add additional information about the item, user and transaction to the meta-features table
projects_columns = ['ProjectSubjectCategoryTree', 'ProjectSubjectSubcategoryTree', 'ProjectResourceCategory', 'ProjectGradeLevelCategory']
donors_columns = ['DonorState', 'DonorCity', 'DonorZip', 'DonorIsTeacher']
schools_columns = ['SchoolMetroType', 'SchoolPercentageFreeLunch', 'SchoolState', 'SchoolCity', 'SchoolZip']
for df_cat, columns, merge_on_column in [(projects, projects_columns, 'ProjectID'), (donors, donors_columns, 'DonorID'), (schools, schools_columns, 'SchoolID')]:
    # Skip merging columns which are already in the table
    columns = np.setdiff1d(columns, meta_items.columns)
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
meta_items['SchoolZip'] = pd.to_numeric(meta_items['SchoolZip'].fillna(-1.).astype(str).str[0:3], downcast='integer')
# Convert teacher status to 1 / 0
meta_items['DonorIsTeacher'] = meta_items['DonorIsTeacher'].map({'Yes': 1, 'No': 0})

# Keep track of all columns which shall be used for training
# However, make sure not to train on features from the item; If otherwise required, simply add projects_columns and schools_columns here
feature_columns_base = set()
feature_columns_base.update(donors_columns)

# Preprocessing: Merge in the information about the date as further meta-features
donations['DonationReceivedDateYear'] = donations['DonationReceivedDate'].dt.year
donations['DonationReceivedDateMonth'] = donations['DonationReceivedDate'].dt.month
donations['DonationReceivedDateDay'] = donations['DonationReceivedDate'].dt.day
donations['DonationReceivedDateDayOfWeek'] = donations['DonationReceivedDate'].dt.dayofweek
donations['DonationReceivedDateTimeOfDay'] = donations['DonationReceivedDate'].dt.hour * 60 + donations['DonationReceivedDate'].dt.minute
date_columns = ['DonationReceivedDateYear', 'DonationReceivedDateMonth', 'DonationReceivedDateDay', 'DonationReceivedDateDayOfWeek', 'DonationReceivedDateTimeOfDay']
meta_items = pd.merge(meta_items, donations[['DonorID', 'ProjectID', *np.setdiff1d(date_columns, meta_items.columns)]], on=['DonorID', 'ProjectID'], how='left', sort=False)
feature_columns_base.update(date_columns)

# Add *IsEqual columns to the meta-features
is_equal_columns = [('ZipIsEqual', 'DonorZip', 'SchoolZip'), ('CityIsEqual', 'DonorCity', 'SchoolCity'), ('StateIsEqual', 'DonorState', 'SchoolState')]
for new_column, c_1, c_2  in is_equal_columns:
    meta_items[new_column] = (meta_items[c_1] == meta_items[c_2]).astype(int)

# Make sure not to train on feature from the item; Otherwise, comment out the following line if necessary
#feature_columns_base.update(list(zip(*is_equal_columns))[0])

# Fill remaining non-numeric and NaN values (e.g. in DonorCity, DonorZip, SchoolCity and SchoolPercentageFreeLunch) with a value otherwise not used: -1
# Most algorithms will not care about NaN (though some are allergic to it) but are keen on converting the data to floats
meta_items[sorted(feature_columns_base)] = meta_items[sorted(feature_columns_base)].apply(pd.to_numeric, errors='coerce').fillna(-1.)

meta_algorithms = {}
meta_algorithms_avail = {'SKLearn-AdaBoostClassifier': ensemble.AdaBoostClassifier, 'SKLearn-BaggingClassifier': ensemble.BaggingClassifier,
    'SKLearn-ExtraTreesClassifier': ensemble.ExtraTreesClassifier, 'SKLearn-GradientBoostingClassifier': ensemble.GradientBoostingClassifier,
    'SKLearn-RandomForestClassifier': ensemble.RandomForestClassifier, 'SKLearn-DecisionTreeClassifier': tree.DecisionTreeClassifier,
    'SKLearn-ExtraTreeClassifier': tree.ExtraTreeClassifier, 'SKLearn-AdaBoostRegressor': ensemble.AdaBoostRegressor,
    'SKLearn-BaggingRegressor': ensemble.BaggingRegressor, 'SKLearn-ExtraTreesRegressor': ensemble.ExtraTreesRegressor,
    'SKLearn-GradientBoostingRegressor': ensemble.GradientBoostingRegressor, 'SKLearn-RandomForestRegressor': ensemble.RandomForestRegressor,
    'SKLearn-DecisionTreeRegressor': tree.DecisionTreeRegressor, 'SKLearn-ExtraTreeRegressor': tree.ExtraTreeRegressor,
    'Keras-NN': NNHelper, 'SKLearn-UserCluster': UserCluster}

# Initialize all selected meta-algorithms
for meta_alg_name, meta_alg_spec in meta_algorithms_args.items():
    if meta_alg_spec['model'] not in meta_algorithms_avail:
        raise ValueError('Expected a valid meta-learner algorithm model from `' + str(sorted(meta_algorithms_avail.keys())) + '` got "' + meta_alg_spec['model'] + '"')

    meta_algorithms[meta_alg_name] = meta_algorithms_avail[meta_alg_spec['model']](**meta_alg_spec['options'])

i = 0
rs = ShuffleSplit(n_splits=n_splits, train_size=train_size, test_size=None)
for train_idx, test_idx in rs.split(meta_items):
    i += 1

    # Preserve which parts of the dataset were used for testing respectively training as to make the final results in the output table meaningful
    meta_items.at[train_idx, 'isTest'] = False
    meta_items.at[test_idx, 'isTest'] = True

    # Preprocessing: Add further information to the meta-features table which is test-train specific
    val_count_columns = ['DonorID']
    for c in val_count_columns:
        # Add the value counts of the train data directly to the table
        value_counts = meta_items.loc[train_idx][c].value_counts(sort=False).reset_index().rename(columns={'index': c, c: 'ValueCounts' + c})
        meta_items.at[train_idx, 'ValueCounts' + c] = pd.merge(meta_items[[c]], value_counts, on=c, how='left', sort=False).loc[train_idx]['ValueCounts' + c]
        # Add the training value counts plus unity to every transaction in the test data as not to reveal how often an item or a user occurs in the test set
        # This leaks information from the training data into the test data, but not the other way around!
        test_value_counts = pd.DataFrame({c: meta_items.loc[test_idx][c].unique(), 'ValueCounts' + c: 1.})
        test_value_counts['ValueCounts' + c] = pd.merge(test_value_counts, value_counts, on=c, how='left', suffixes=('_x', '_y'), sort=False)[['ValueCounts' + c + '_x', 'ValueCounts' + c + '_y']].sum(axis=1)
        meta_items.at[test_idx, 'ValueCounts' + c] = pd.merge(meta_items[[c]], test_value_counts, on=c, how='left', sort=False).loc[test_idx]['ValueCounts' + c]

    feature_columns_base.update(['ValueCounts' + c for c in val_count_columns])

    val_counts_by_user_columns = ['ProjectID']
    for c in val_counts_by_user_columns:
        value_counts = meta_items.loc[train_idx][c].value_counts(sort=False).reset_index().rename(columns={'index': c, c: 'ValueCountsByUser' + c})
        value_counts_by_user = pd.merge(meta_items[['DonorID', c]], value_counts, on=c, how='left', sort=False).groupby('DonorID')['ValueCountsByUser' + c].sum().reset_index()
        meta_items['ValueCountsByUser' + c] = pd.merge(meta_items[['DonorID']], value_counts_by_user, on='DonorID', how='left', sort=False)['ValueCountsByUser' + c]
        # Fill the remaining user-values which were not covered by the training set with the mode of the train set
        meta_items['ValueCountsByUser' + c] = meta_items['ValueCountsByUser' + c].fillna(meta_items.loc[train_idx]['ValueCountsByUser' + c].mode()[0])

    feature_columns_base.update(['ValueCountsByUser' + c for c in val_counts_by_user_columns])

    user_mean_columns = np.concatenate((['DonationAmount'], projects_columns, schools_columns, list(zip(*is_equal_columns))[0]))
    for c in user_mean_columns:
        aggregated_mean = meta_items.loc[train_idx].groupby('DonorID')[c].mean().reset_index().rename(columns={c: 'UserMean' + c})
        meta_items['UserMean' + c] = pd.merge(meta_items[['DonorID']], aggregated_mean, on='DonorID', how='left', sort=False)['UserMean' + c]
        # Fill the remaining user-values which were not covered by the training set with the mean of the train set
        meta_items['UserMean' + c] = meta_items['UserMean' + c].fillna(meta_items.loc[train_idx][c].mean())

    feature_columns_base.update(['UserMean' + c for c in user_mean_columns])

    for acc_name in algorithms_accuracy_name:
        # Convert between column names
        alg_acc_to_alg_columns = {acc_name + alg_name: alg_name for alg_name in algorithms_name}

        alg_comparison = meta_items.loc[test_idx][sorted(alg_acc_to_alg_columns.keys())].mean(axis=0)
        best_alg_idx = alg_comparison.idxmin()
        best_alg = alg_acc_to_alg_columns[best_alg_idx], alg_comparison[best_alg_idx]
        combined_best = 'combined', meta_items.loc[test_idx][sorted(alg_acc_to_alg_columns.keys())].min(axis=1).mean()

        logger.info('{acc_name:<16s} ({:^15s}) (shuffle {i:>d}/{n_splits:<d}) :: | {:^25s} | Test-{acc_name}: {:>7.2f}'.format('overall best', *best_alg, i=i, n_splits=n_splits, acc_name=acc_name))
        logger.info('{acc_name:<16s} ({:^15s}) (shuffle {i:>d}/{n_splits:<d}) :: | {:^25s} | Test-{acc_name}: {:>7.2f}'.format('combined best', *combined_best, i=i, n_splits=n_splits, acc_name=acc_name))
        logger.debug('{acc_name:<16s} ({:^15s}) (shuffle {i:>d}/{n_splits:<d}) :: | {:^25s} | Value-Counts: {:<50s}'.format('combined best', combined_best[0], str(meta_items.loc[test_idx][sorted(alg_acc_to_alg_columns.keys())].idxmin(axis=1).value_counts().to_dict()), i=i, n_splits=n_splits, acc_name=acc_name))

        for meta_alg_name, meta_alg in sorted(meta_algorithms.items(), key=lambda x: x[0]):
            # Preserve the initial feature columns but allow algorithms to amend a newly spawned instance of the set prior to fitting
            feature_columns = feature_columns_base.copy()
            # Add the predictions of algorithms from the learning subsystem to the feature-columns if a stacking meta-learner is explicitly enabled
            if 'stacking' in meta_algorithms_args[meta_alg_name] and meta_algorithms_args[meta_alg_name]['stacking'] is True:
                feature_columns.update(['Prediction' + alg_name for alg_name in algorithms_name])

            if 'accuracy prediction' in meta_algorithms_args[meta_alg_name]['methods']:
                for alg_name in algorithms_name:
                    meta_alg.fit(meta_items.loc[train_idx][sorted(feature_columns)].values, meta_items.loc[train_idx][acc_name + alg_name].values)
                    meta_items['Prediction' + meta_alg_name + acc_name + alg_name] = meta_alg.predict(meta_items[sorted(feature_columns)].values)

                    log_line = '{meta_alg_name:<35s} ({:^15s}) (shuffle {i:>d}/{n_splits:<d}) {acc_name:<16s}{alg_name:<25s} ::'.format('meta acc', i=i, n_splits=n_splits, acc_name=acc_name, alg_name=alg_name, meta_alg_name=meta_alg_name)
                    for eval_acc_name, eval_acc_method in sorted({'MAE': mean_absolute_error, 'RMSE': lambda x, y: np.sqrt(mean_squared_error(x, y))}.items(), key=lambda x: x[0]):
                        train_acc = eval_acc_method(meta_items.loc[train_idx]['Prediction' + meta_alg_name + acc_name + alg_name].values, meta_items.loc[train_idx][acc_name + alg_name].values)
                        test_acc = eval_acc_method(meta_items.loc[test_idx]['Prediction' + meta_alg_name + acc_name + alg_name].values, meta_items.loc[test_idx][acc_name + alg_name].values)
                        log_line += ' | Training-{0:s}: {train_acc:>7.2f}, Test-{0:s}: {test_acc:>7.2f}'.format(eval_acc_name, train_acc=train_acc, test_acc=test_acc)

                    logger.debug(log_line)

                meta_alg_to_alg_acc_columns = {'Prediction' + meta_alg_name + acc_name + alg_name: acc_name + alg_name for alg_name in algorithms_name}
                # Look up the lowest prediction made by the meta-learning algorithm and select the column which contains the actual result
                predict_alg_acc_columns = meta_items[sorted(meta_alg_to_alg_acc_columns.keys())].idxmin(axis=1).astype('category').cat.rename_categories(meta_alg_to_alg_acc_columns)

                meta_items['MetaPrediction' + meta_alg_name] = meta_items.lookup(meta_items.index, predict_alg_acc_columns)
                best_meta = 'meta accuracy', meta_items.loc[train_idx]['MetaPrediction' + meta_alg_name].mean(), meta_items.loc[test_idx]['MetaPrediction' + meta_alg_name].mean()
                logger.info('{meta_alg_name:<35s} ({:^15s}) (shuffle {i:>d}/{n_splits:<d}) :: | {:^25s} | Training-{acc_name}: {:>7.2f}, Test-{acc_name}: {:>7.2f}'.format('meta acc', *best_meta, i=i, n_splits=n_splits, acc_name=acc_name, meta_alg_name=meta_alg_name))

            if 'classification' in meta_algorithms_args[meta_alg_name]['methods']:
                # Treat the best performing algorithm of a row as the row's class
                meta_items['SubalgorithmCategory'] = meta_items[sorted(alg_acc_to_alg_columns.keys())].idxmin(axis=1)
                # Assign category IDs in descending order with the average algorithm training performance as to make the distances interpretable
                sub_alg_value_counts = meta_items.loc[train_idx]['SubalgorithmCategory'].value_counts()
                sub_alg_sorted = sorted(alg_acc_to_alg_columns.keys(), key=lambda x: -1 * sub_alg_value_counts[x])
                meta_items['SubalgorithmCategory'] = meta_items['SubalgorithmCategory'].astype(pd.api.types.CategoricalDtype(categories=sub_alg_sorted))

                sample_weight_mat = np.sort(meta_items.loc[train_idx][sorted(alg_acc_to_alg_columns.keys())], axis=1)
                sample_weight = np.power(class_penalty_base, (sample_weight_mat[:, 1] - sample_weight_mat[:, 0]) / (np.max(sample_weight_mat[:, -1], axis=0) - np.min(sample_weight_mat[:, 0], axis=0)))
                meta_alg.fit(meta_items.loc[train_idx][sorted(feature_columns)].values, meta_items.loc[train_idx]['SubalgorithmCategory'].cat.codes.values, sample_weight=sample_weight)
                meta_items['SubalgorithmPrediction' + meta_alg_name + acc_name] = pd.Categorical.from_codes(meta_alg.predict(meta_items[sorted(feature_columns)].values), categories=sub_alg_sorted)

                logger.debug('{meta_alg_name:<35s} ({:^15s}) (shuffle {i:>d}/{n_splits:<d}) {acc_name:<16s} :: | Value-Counts: {:<50s}'.format('meta class', str(meta_items.loc[test_idx]['SubalgorithmPrediction' + meta_alg_name + acc_name].value_counts().to_dict()), i=i, n_splits=n_splits, acc_name=acc_name, meta_alg_name=meta_alg_name))
                classification_accuracy = (meta_items.loc[test_idx]['SubalgorithmCategory'] == meta_items.loc[test_idx]['SubalgorithmPrediction' + meta_alg_name + acc_name]).mean()
                logger.debug('{meta_alg_name:<35s} ({:^15s}) (shuffle {i:>d}/{n_splits:<d}) {acc_name:<16s} :: | Test-Accuracy: {:>7.2%}'.format('meta class', classification_accuracy, i=i, n_splits=n_splits, acc_name=acc_name, meta_alg_name=meta_alg_name))

                meta_items['MetaSubalgorithmPrediction' + meta_alg_name] = meta_items.lookup(meta_items.index, meta_items['SubalgorithmPrediction' + meta_alg_name + acc_name])
                best_meta = 'meta classify', meta_items.loc[train_idx]['MetaSubalgorithmPrediction' + meta_alg_name].mean(), meta_items.loc[test_idx]['MetaSubalgorithmPrediction' + meta_alg_name].mean()
                logger.info('{meta_alg_name:<35s} ({:^15s}) (shuffle {i:>d}/{n_splits:<d}) :: | {:^25s} | Training-{acc_name}: {:>7.2f}, Test-{acc_name}: {:>7.2f}'.format('meta class', *best_meta, i=i, n_splits=n_splits, acc_name=acc_name, meta_alg_name=meta_alg_name))

if output_filepath is not None and output_filepath is not False:
    meta_items.to_csv(output_filepath)
