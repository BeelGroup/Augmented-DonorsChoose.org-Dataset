# Code and Literature Repository for Investigating Meta-Learning Algorithms in the Context of Recommender Systems

## Inspiration

The main inspiration for this research is based on the work performed by the ADAPT group at the Trinity college in Dublin. Of special note for this project is the research outlined in the paper [One-at-a-time: A Meta-Learning Recommender-System for Recommendation-Algorithm Selection on Micro Level](https://arxiv.org/abs/1805.12118).

## Abstract

The DonorsChoose.org dataset of past donations provides a big and feature-rich corpus of user and item matches. The dataset matches donors to projects in which they might be interested in and hence is intrinsically about recommendations. Due to the availability of item-, user- and transaction-features, this corpus represents a suitable candidate for meta-learning approaches to be tested. This study aims at providing an augmented corpus for further recommender systems studies to test and evaluate meta-learning approaches. In our augmentation, metadata of collaborative and content-based filtering techniques is amended to the corpus. It is further extended with aggregated statistics for users and transactions and an exemplary meta-learning experiment. The performance in the learning subsystem is measured via the recall of recommended items in a Top-N test set. The augmented dataset and the source code are released into the public domain at GitHub:BeelGroup/Augmented-DonorsChoose.org-Dataset.

## Corpus Features and Augmented Metadata

The following represents an organized list of columns with each being present for every entry in the itemized transaction table.

### Transaction

* `DonationID`
* `DonationIncludedOptionalDonation`
* `DonationReceivedDate`
  * `DonationReceivedDateDay`
  * `DonationReceivedDateDayOfWeek`
  * `DonationReceivedDateMonth`
  * `DonationReceivedDateTimeOfDay`
  * `DonationReceivedDateYear`
* `DonorCartSequence`
* `DonorID`
* `ProjectID`
* `DonationAmount`
* `(Zip|City|Sate)IsEqual`
  i.e. whether user and item have identical values

### User

* `DonorState`
* `DonorCity`
* `DonorZip`
* `DonorIsTeacher`
* `Concat[(DonorState|DonorCity|...)+]`
  i.e. the concatenated value of mentioned columns

### Item

* `SchoolID`
* `ProjectGradeLevelCategory`
* `ProjectResourceCategory`
* `ProjectSubjectCategoryTree`
* `ProjectSubjectSubcategoryTree`
* `SchoolCity`
* `SchoolMetroType`
* `SchoolPercentageFreeLunch`
* `SchoolState`
* `SchoolZip`

### Learning Subsystem

* Collaborative Filtering Techniques
  * `AbsoluteErrorSKLearn-(KNN|SVD)`
  * `SquareErrorSKLearn-(KNN|SVD)`
  * `RecallAtPositionSKLearn-(KNN|SVD)`
  * `PredictionSKLearn-(KNN|SVD)`
    i.e. decomposition of the matrix or the interactions of the neighbor
* Content-based Filtering Techniques
  * `RecallAtPosition(FastText|Tfidf)`
  * `Prediction(FastText|Tfidf)`
    i.e. cosine similarity of user profile and recommendation
* Collaborative recommendations for user-groups
  * `AbsoluteErrorGroupBy[(DonorState|DonorCity|...)+]-SKLearn-SVD`
  * `SquareErrorGroupBy[(DonorState|DonorCity|...)+]-SKLearn-SVD`
  * `RecallAtPositionGroupBy[(DonorState|DonorCity|...)+]-SKLearn-SVD`
  * `PredictionGroupByDonor[(DonorState|DonorCity|...)+]-SKLearn-SVD`

### Statistics

* General
  * `isTest`
    i.e. whether the entry was used for testing during the holdout split
* Values aggregated by User
  * `ValueCountsDonorID`
    i.e. number of transactions
  * `ValueCountsByUserProjectID`
    i.e. whether the user donated to popular projects
  * `UserMean(DonationAmount|ProjectGradeLevelCategory|...)`

### Meta-Learning System

* `MetaPrediction(BaggingRg|GradientBoostingRg|...)RecallAtPosition(SKLearn-SVD|FastText|...)`
  i.e. prediction of the error of the individual meta-learners in the error prediction step
* `MetaPrediction(BaggingRg|GradientBoostingRg|...)RecallAtPosition(SKLearn-SVD|FastText|...)`
  i.e. prediction if the suggested algorithm is selected via error prediction
* `SubalgorithmPrediction(BaggingRg|GradientBoostingRg|...)RecallAtPosition`
  i.e. prediction of the class in the classification step
* `MetaSubalgorithmPrediction(BaggingRg|GradientBoostingRg|...)RecallAtPosition(SKLearn-SVD|FastText|...)`
  i.e. prediction if the suggested algorithm is selected via classification
* `SubalgorithmCategory`
  i.e. 'category' of the transaction if assigned to the best performing algorithm

## Code Design

This repository is the single source of truth for the whole scientific exploration of the augmentation and evaluation of the DonorsChoose.org dataset. In addition to the actual code needed for reproduction, the repository contains all relevant status updates. The dedicated folder for documentation is appropriately named `doc`. The dataset may be stored in `data`. Changes happening within this folder are ignored by the version control system. The main programs are `learners.py` and `meta-learners.py` with helper functions being outsource to `recsys`. The first python-script is dedicated to creating a dataset augmented with results from various filtering techniques. It represents the learning subsystem and performs the computationally most expensive steps. The second python-script further augments the dataset and executes the meta-learning algorithms.

### Learning Subsystem

The learning subsystem is contained in the `learners.py` script and is adaptable via the configuration file `config.yml`. Most parameters should be self-explanatory. If deemed necessary, a small explanatory string is added. Most notably is the dictionary `algorithms_args` which specifies all the algorithms from the learning subsystem which to execute on the data.

### Meta-learning System

The final augmentation is performed in `meta-learners.py`. The program's behavior can be configured via `config-meta-learners.yml`. Its most important option is the dictionary describing the meta-algorithms which to execute on the dataset.

## Code Snippets

### Configuration of Visuals

* Non-interactive plotting

```python
import matplotlib as mpl

mpl.use('cairo')

import matplotlib.pyplot as plt
```

* Prettify plots

```python
import seaborn as sns

sns.set_style('whitegrid')
```

* Suitable Aspect Ratio for Plots

```python
mpl.rcParams['figure.figsize'] = 6.4, 3.2
```

* Enforce text rendering via LaTeX and mimic the font of the default matplotlib text

```python
plt.rc('text', usetex=True)
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
```

### Visualizations

* Donated amount in bins on a logarithmic scale

```python
items_orig = donations[['ProjectID', 'DonorID', 'DonationAmount']]

plt.figure()
plt.hist(items_orig['DonationAmount'], bins=np.logspace(np.log10(items_orig['DonationAmount'].min()), np.log10(items_orig['DonationAmount'].max()), num=28 + 1), histtype='step')
plt.gca().set_xscale('log')
plt.xlabel('Donated Amount')
plt.ylabel('#Occurrence')
plt.tight_layout()
plt.savefig('DonationAmount - Distribution of the donated amount on a logarithmic scale.pdf', bbox_inches='tight')
plt.close()
```

* Donated amount in bins on a logarithmic scale for clean subset

```python
items_orig = donations.groupby(['DonorID', 'ProjectID'])['DonationAmount'].sum().reset_index()
# Perform preliminary data cleaning
items_orig = items_orig.drop(items_orig.query('0. <= DonationAmount <= 2.').index)
value_counts = items_orig['DonorID'].value_counts()
items_orig = items_orig[items_orig['DonorID'].isin(value_counts.index[value_counts >= 2])]

plt.figure()
plt.hist(items_orig['DonationAmount'], bins=np.logspace(np.log10(items_orig['DonationAmount'].min()), np.log10(items_orig['DonationAmount'].max()), num=13 + 1), density=True, histtype='step')
plt.gca().set_xscale('log')
plt.xlabel('Donated Amount')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('DonationAmount - Distribution of the donated amount on a logarithmic scale (for donors with at least 2 donations, excluding duplicates and low donations).pdf', bbox_inches='tight')
plt.close()
```

* Distribution of ratings

```python
# Shrink size and enlarge font
mpl.rcParams['figure.figsize'][0] /= 1.3

plt.figure()
plt.grid(b=False, axis='x')

plt.hist(items['DonationAmount'], bins=5, density=True, histtype='step')
plt.xticks([1.45, 2.2, 3., 3.8, 4.6], np.arange(1, 5+1))
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('DonationAmount - Distribution of ratings for logarithmic bins and excluded outliers.pdf', bbox_inches='tight')
plt.close()

mpl.rcParams['figure.figsize'][0] *= 1.3
```

* Number of user donations

```python
# Shrink size and enlarge font
mpl.rcParams['figure.figsize'][0] /= 1.3

plt.figure()
plt.grid(b=False, axis='x')

user_value_counts = items['DonorID'].value_counts()
# Disregard outliers
user_value_counts = user_value_counts[user_value_counts <= user_value_counts.mean() + user_value_counts.std()]

plt.hist(user_value_counts, bins=30, density=True, histtype='step')

plt.xlabel('Interactions per user')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('DonorID - Distribution of number of interactions per user.pdf', bbox_inches='tight')
plt.close()

mpl.rcParams['figure.figsize'][0] *= 1.3
```

* RMSE for collaborative filtering techniques

```python
plt.figure()
plt.grid(b=False, axis='x')

algorithms_name = ['zero', 'mean', 'random', 'SKLearn-KNN', 'SKLearn-NMF', 'SKLearn-SVD', 'SciPy-SVD']
average_rmse = [np.sqrt(np.square(np.zeros(items.shape[0]) - items['DonationAmount']).mean()),
  np.sqrt(np.square(np.full(items.shape[0], items['DonationAmount'].mean()) - items['DonationAmount']).mean()),
  np.sqrt(np.square(np.random.uniform(low=min(items['DonationAmount']), high=max(items['DonationAmount']), size=items.shape[0]) - items['DonationAmount']).mean()),
  np.sqrt(items['SquareErrorSKLearn-KNN'].mean()),
  np.sqrt(items['SquareErrorSKLearn-NMF'].mean()),
  np.sqrt(items['SquareErrorSKLearn-SVD'].mean()),
  np.sqrt(items['SquareErrorSciPy-SVD'].mean())]

plt.errorbar(np.arange(len(average_rmse)), average_rmse, xerr=0.45, markersize=0., ls='none')

plt.xticks(np.arange(len(algorithms_name)), algorithms_name)

plt.xlabel('Algorithm')
plt.ylabel('Test RMSE')

plt.gcf().autofmt_xdate()
plt.tight_layout()

plt.savefig('Collaborative Filters - RMSE for DIY algorithms and some baselines.pdf', bbox_inches='tight')
plt.close()
```

* Recall@N for collaborative and content-based filters

```python
plt.figure()
plt.grid(b=False, axis='x')

algorithms_name = ['SKLearn-KNN', 'SKLearn-NMF', 'SKLearn-SVD', 'SciPy-SVD', 'Tfidf']
algorithms_pretty_name = ['SKLearn-KNN', 'SKLearn-NMF', 'SKLearn-SVD', 'SciPy-SVD', 'SKLearn-TF-IDF']
average_recall = [items['RecallAtPosition' + alg_name].mean() for alg_name in algorithms_name]

plt.errorbar(np.arange(len(average_recall)), average_recall, xerr=0.45, markersize=0., ls='none')

plt.xticks(np.arange(len(algorithms_pretty_name)), algorithms_pretty_name)
plt.ylim(ymin=-1)

plt.xlabel('Algorithm')
plt.ylabel('Average position in Top-N test set')

plt.gcf().autofmt_xdate()
plt.tight_layout()

plt.savefig('Collaborative and Content-based Filters - Average position in Top-N test set for various algorithms.pdf', bbox_inches='tight')
plt.close()
```

```python
plt.figure()
plt.grid(b=False, axis='x')

algorithms_name = ['SKLearn-KNN', 'SKLearn-SVD', 'Tfidf']
algorithms_pretty_name = ['SKLearn-KNN', 'SKLearn-SVD', 'SKLearn-TF-IDF']

plt.hist([items['RecallAtPosition' + alg_name] for alg_name in algorithms_name], bins=10, density=True, label=algorithms_pretty_name, histtype='step')

plt.legend(loc=9)
plt.xlabel('Position in Top-N test set')
plt.ylabel('Frequency')

plt.tight_layout()

plt.savefig('Collaborative and Content-based Filters - Distribution of position in Top-N test set for various algorithms.pdf', bbox_inches='tight')
plt.close()
```

* Learning subsystem Recall@N performance

```python
plt.figure()
plt.grid(b=False, axis='x')

algorithms_name = ['SKLearn-KNN', 'SKLearn-SVD', 'Tfidf', 'FastText']
recall_pos = [items['RecallAtPosition' + alg_name].values for alg_name in algorithms_name] + [items[['RecallAtPosition' + alg_name for alg_name in algorithms_name]].min(axis=1).values]
algorithms_pretty_name = ['KNN', 'SVD', 'TF-IDF', 'FastText', 'Combined']

plt.boxplot(recall_pos, positions=np.arange(len(algorithms_pretty_name)), meanline=True, showmeans=True, showfliers=False)

plt.xticks(np.arange(len(algorithms_pretty_name)), algorithms_pretty_name)
plt.ylim(ymin=-1)

plt.xlabel('Algorithm')
plt.ylabel('Position in Top-N test set')

plt.gcf().autofmt_xdate()
plt.tight_layout()

plt.savefig('Learning subsystem - Position in Top-N test set for various algorithms.pdf', bbox_inches='tight')
plt.close()
```

```python
plt.figure()
plt.grid(b=False, axis='x')

algorithms_name = ['SKLearn-KNN', 'SKLearn-SVD', 'Tfidf', 'FastText']
algorithms_pretty_name = {'SKLearn-KNN': 'KNN', 'SKLearn-SVD': 'SVD', 'Tfidf': 'TF-IDF', 'FastText': 'FastText'}
algorithms_value_counts = items[['RecallAtPosition' + alg_name for alg_name in algorithms_name]].idxmin(axis=1).value_counts().rename(dict(zip(['RecallAtPosition' + alg_name for alg_name in algorithms_name], algorithms_name))).to_dict()

plt.hist([items['RecallAtPosition' + alg_name] for alg_name in algorithms_name], bins=10, density=True, label=['{:<s} ({:<2.2%} overall best)'.format(algorithms_pretty_name[alg_name], algorithms_value_counts[alg_name] / items.shape[0]) for alg_name in algorithms_name], histtype='step')

plt.legend(loc=9)
plt.xlabel('Position in Top-N test set')
plt.ylabel('Frequency')

plt.tight_layout()

plt.savefig('Learning subsystem - Distribution of position in Top-N test set for various algorithms.pdf', bbox_inches='tight')
plt.close()
```

* Meta-learner performance for classification and error prediction

```python
meta_subset = meta_items.loc[test_idx]

plt.figure()
plt.grid(b=False, axis='x')

meta_algorithms_name = [('Bagging', 'Bagging'), ('DecisionTree', 'DecisionTree'), ('BalancedDecisionTree', 'BalancedDTree'), ('GradientBoosting', 'GradientBoosting'), ('NeuralNetwork', 'NeuralNetwork')]
algorithm_selection_columns = [('MetaSubalgorithmPrediction', 'CL'), ('MetaPrediction', 'EP')]
meta_algorithms_column = np.array([[pre[0] + meta_alg_name[0] for pre in algorithm_selection_columns] for meta_alg_name in meta_algorithms_name]).flatten()
meta_algorithms_pretty_name = np.array([[pre[1] + ' ' + meta_alg_name[1] for pre in algorithm_selection_columns] for meta_alg_name in meta_algorithms_name]).flatten()
average_recall = [meta_subset[c].mean() for c in meta_algorithms_column]

plt.errorbar(np.arange(len(average_recall)), average_recall, color=np.array([[c for _ in range(len(algorithm_selection_columns))] for c in plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(meta_algorithms_name)]]).flatten(), xerr=0.45, markersize=0., ls='none')
plt.axhline(y=meta_subset.lookup(meta_subset.index, meta_subset['SubalgorithmCategory']).mean(), color='orange', linestyle='--')

plt.xticks(np.arange(len(meta_algorithms_pretty_name)), meta_algorithms_pretty_name)
plt.ylim(ymin=-1)

plt.xlabel('Algorithm')
plt.ylabel('Average position in Top-N test set')

plt.gcf().autofmt_xdate()
plt.tight_layout()

plt.savefig('Meta-learner as Classifier and Error Predictor - Average position in Top-N test set for various meta-learner algorithms.pdf', bbox_inches='tight')
plt.close()
```

* Learning subsystem Recall@N performance with augmented filtering techniques

```python
plt.figure()
plt.grid(b=False, axis='x')

algorithms_name = ['SKLearn-KNN', 'SKLearn-SVD', 'GroupByDonorStateCityZip-SKLearn-SVD', 'GroupByDonorStateCity-SKLearn-SVD', 'Tfidf', 'FastText']
recall_pos = [items['RecallAtPosition' + alg_name].values for alg_name in algorithms_name] + [items[['RecallAtPosition' + alg_name for alg_name in algorithms_name]].min(axis=1).values]
algorithms_value_counts = items[['RecallAtPosition' + alg_name for alg_name in algorithms_name]].idxmin(axis=1).value_counts().rename(dict(zip(['RecallAtPosition' + alg_name for alg_name in algorithms_name], algorithms_name))).to_dict()

algorithms_name = algorithms_name + ['Combined']
algorithms_value_counts['Combined'] = items.shape[0]
algorithms_pretty_name = {'SKLearn-KNN': 'KNN', 'SKLearn-SVD': 'SVD', 'GroupByDonorStateCityZip-SKLearn-SVD': 'SVD (State, City, Zip)', 'GroupByDonorStateCity-SKLearn-SVD': 'SVD (State, City)', 'Tfidf': 'TF-IDF', 'FastText': 'FastText', 'Combined': 'Combined'}

plt.boxplot(recall_pos, positions=np.arange(len(algorithms_pretty_name)), meanline=True, showmeans=True, showfliers=False)

# This got a little bit out of hand...
# Actually just the percentage of each algorithm's contribution in the combined best is printed in a smaller font below the algorithm's name
plt.xticks(np.arange(len(algorithms_pretty_name)), [r'{{\fontsize{{1em}}{{3em}}\selectfont{{}}{0:<s}}}{1}{{\fontsize{{0.8em}}{{3em}}\selectfont{{}}{2:<2.2f}\%}}'.format(algorithms_pretty_name[alg_name], '\n', 100 * algorithms_value_counts[alg_name]  / items.shape[0]) for alg_name in algorithms_name])
plt.ylim(ymin=-1)

plt.xlabel('Algorithm')
plt.ylabel('Position in Top-N test set')

plt.gcf().autofmt_xdate()
plt.tight_layout()

plt.savefig('Learning subsystem - Position in Top-N test set for various algorithms with augmented filtering techniques.pdf', bbox_inches='tight')
plt.close()
```

* Learning subsystem Recall@N performance

```python
plt.figure()
plt.grid(b=False, axis='x')

algorithms_name = ['SKLearn-KNN', 'SKLearn-SVD', 'Tfidf', 'FastText']
recall_pos = [items['RecallAtPosition' + alg_name].values for alg_name in algorithms_name]
algorithms_value_counts = items[['RecallAtPosition' + alg_name for alg_name in sorted(algorithms_name)]].idxmin(axis=1).value_counts().rename(dict(zip(['RecallAtPosition' + alg_name for alg_name in algorithms_name], algorithms_name))).to_dict()

algorithms_pretty_name = {'SKLearn-KNN': 'KNN', 'SKLearn-SVD': 'SVD', 'GroupByDonorStateCityZip-SKLearn-SVD': 'SVD (State, City, Zip)', 'GroupByDonorStateCity-SKLearn-SVD': 'SVD (State, City)', 'Tfidf': 'TF-IDF', 'FastText': 'FastText', 'Combined': 'Combined'}

plt.boxplot(recall_pos, positions=np.arange(len(algorithms_name)), meanline=True, showmeans=True, showfliers=False)

# This got a little bit out of hand...
# Actually just the percentage of each algorithm's contribution in the combined best is printed in a smaller font below the algorithm's name
plt.xticks(np.arange(len(algorithms_name)), [r'{{\fontsize{{1em}}{{3em}}\selectfont{{}}{0:<s}}}{1}{{\fontsize{{0.8em}}{{3em}}\selectfont{{}}{2:<2.2f}\%}}'.format(algorithms_pretty_name[alg_name], '\n', 100 * algorithms_value_counts[alg_name]  / items.shape[0]) for alg_name in algorithms_name])
plt.ylim(ymin=-1)

plt.xlabel('Algorithm')
plt.ylabel('Position in Top-N test set')

plt.gcf().autofmt_xdate()
plt.tight_layout()

plt.savefig('Learning subsystem - Position in Top-N test set.pdf', bbox_inches='tight')
plt.close()
```

* Meta-learner performance for classification and error prediction with augmented learning subsystem filtering techniques

```python
meta_subset = meta_items.loc[test_idx]

plt.figure()
plt.grid(b=False, axis='x')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

meta_algorithms_name = [('MetaSubalgorithmPredictionBaggingCl', 'CL Bagging', colors[0]), ('MetaPredictionBaggingRg', 'EP Bagging', colors[3]), ('MetaSubalgorithmPredictionDecisionTreeCl', 'CL Decision Tree', colors[0]), ('MetaPredictionDecisionTreeRg40', 'EP Decision Tree', colors[3]), ('MetaSubalgorithmPredictionUserClusterKMeans', 'User-Clustering', colors[0]), ('MetaPredictionGradientBoostingRg', 'EP Gradient Boosting', colors[3]), ('MetaSubalgorithmPredictionStackingDecisionTree', 'Stacking DTree', colors[4])]
average_recall = [meta_subset[c].mean() for c in list(zip(*meta_algorithms_name))[0]]

plt.errorbar(np.arange(len(average_recall)), average_recall, color=list(zip(*meta_algorithms_name))[2], xerr=0.45, markersize=0., ls='none')
plt.axhline(y=meta_subset[meta_subset['SubalgorithmCategory'].mode()[0]].mean(), color='orange', linestyle='--')
plt.axhline(y=meta_subset.lookup(meta_subset.index, meta_subset['SubalgorithmCategory']).mean(), color='orange', linestyle='-')

plt.xticks(np.arange(len(meta_algorithms_name)), list(zip(*meta_algorithms_name))[1])
plt.ylim(ymin=-1)

plt.xlabel('Algorithm')
plt.ylabel('Average position in Top-N test set')

plt.gcf().autofmt_xdate()
plt.tight_layout()

plt.savefig('Meta-learner as Classifier and Error Predictor - Average position in Top-N test set for various meta-learner algorithms with augmented learning subsystem filtering techniques.pdf', bbox_inches='tight')
plt.close()
```

* Meta-learner performance

```python
meta_subset = meta_items.loc[test_idx]

plt.figure()
plt.grid(b=False, axis='x')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

meta_algorithms_name = [('MetaSubalgorithmPredictionDecisionTreeRg', 'Classifying DTree', colors[0]), ('MetaPredictionGradientBoostingRg', 'GB Recall Prediction', colors[1]), ('MetaSubalgorithmPredictionUserClusterKMeans', 'K-Means Clustering', colors[2]), ('MetaSubalgorithmPredictionStackingDecisionTree', 'Stacking DTree', colors[3])]
average_recall = [meta_subset[c].mean() for c in list(zip(*meta_algorithms_name))[0]]

plt.errorbar(np.arange(len(average_recall)), average_recall, color=list(zip(*meta_algorithms_name))[2], xerr=0.45, markersize=0., ls='none')
plt.axhline(y=meta_subset[meta_subset['SubalgorithmCategory'].mode()[0]].mean(), color='orange', linestyle='--')

plt.xticks(np.arange(len(meta_algorithms_name)), list(zip(*meta_algorithms_name))[1])
plt.ylim(ymin=meta_subset.lookup(meta_subset.index, meta_subset['SubalgorithmCategory']).mean()-1)

plt.xlabel('Algorithm')
plt.ylabel('Average position in Top-N test set')

plt.gcf().autofmt_xdate()
plt.tight_layout()

plt.savefig('Meta-learner Performance - Average position in Top-N test set for various meta-learner algorithms with augmented learning subsystem filtering techniques.pdf', bbox_inches='tight')
plt.close()
```

## Past Roadmap

* Find a suitable dataset for meta-learning
  * Candidates should provide information about the user, the item and about the context of each transaction
  * Adequate sources might be [kaggle](https://www.kaggle.com), [Google public datasets](https://cloud.google.com/public-datasets/) and previous [RecSys challenges](https://recsys.acm.org/)
* Evaluate existing software frameworks for their applicability as meta-feature generators
  * Meta-feature algorithms should include collaborative, content based and possibly deep learning based approaches
  * Suitable frameworks might be Tensorflow, scikit-learn and higher level libraries like Keras and scikit-surprise
* Train and compare various meta-learning models
  * Predict either rating error or reformulate algorithm selection as classification problem
  * Evaluate model using appropriate variables, possible candidates might be the normalized discounted cumulative gain or the Kendall rank correlation coefficient

## Outlook

* Decaying rating based on the date of the donation
* Use average algorithm with lowest overall error for each cluster in the user-clustering approach
* Algorithm Selection as ranking task using Meta-Learning
