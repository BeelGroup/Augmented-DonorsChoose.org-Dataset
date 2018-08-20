# Code and Literature Repository for Investigating Meta-Learning Algorithms in the Context of Recommender Systems

## Inspiration

The main inspiration for this research is based on the work performed by the ADAPT group at the Trinity college in Dublin. Of special note for this project is the research outlined in the paper [One-at-a-time: A Meta-Learning Recommender-System for Recommendation-Algorithm Selection on Micro Level](https://arxiv.org/abs/1805.12118).

## Roadmap

* Find a suitable dataset for meta-learning
  * Candidates should provide information about the user, the item and about the context of each transaction
  * Adequate sources might be [kaggle](https://www.kaggle.com), [Google public datasets](https://cloud.google.com/public-datasets/) and previous [RecSys challenges](https://recsys.acm.org/)
* Evaluate existing software frameworks for their applicability as meta-feature generators
  * Meta-feature algorithms should include collaborative, content based and possibly deep learning based approaches
  * Suitable frameworks might be Tensorflow, scikit-learn and higher level libraries like Keras and scikit-surprise
* Train and compare various meta-learning models
  * Predict either rating error or reformulate algorithm selection as classification problem
  * Evaluate model using appropriate variables, possible candidates might be the normalized discounted cumulative gain or the Kendall rank correlation coefficient

## Code Snippets

### Non-interactive plotting

```python
import matplotlib as mpl

mpl.use('cairo')

import matplotlib.pyplot as plt
```

### Visualizations

* Plot the donated amount via bins on a logarithmic scale

```python
items_orig = pd.merge(donations[['ProjectID', 'DonorID', 'DonationAmount']], projects[['ProjectID', 'SchoolID']], on='ProjectID', how='inner', sort=False)

plt.figure()
plt.hist(items_orig['DonationAmount'], bins=np.logspace(np.log10(items_orig['DonationAmount'].min()), np.log10(items_orig['DonationAmount'].max()), num=28 + 1), histtype='step')
plt.gca().set_xscale('log')
plt.xlabel('Donated Amount')
plt.ylabel('#Occurrence')
plt.tight_layout()
plt.savefig('DonationAmount - Distribution of the donated amount on a logarithmic scale.pdf')
plt.close()
```

* Plot the donated amount via bins on a logarithmic scale for clean subset

```python
items_orig = pd.merge(donations[['ProjectID', 'DonorID', 'DonationAmount']], projects[['ProjectID', 'SchoolID']], on='ProjectID', how='inner', sort=False)
# Perform preliminary data cleaning
items_orig = items_orig.drop_duplicates(['DonorID', 'SchoolID'], keep='first')
items_orig = items_orig.drop(items_orig.query('0. <= DonationAmount <= 2.').index)
value_counts = items_orig['DonorID'].value_counts()
items_orig = items_orig[items_orig['DonorID'].isin(value_counts.index[value_counts >= 2])]

plt.figure()
plt.hist(items_orig['DonationAmount'], bins=np.logspace(np.log10(items_orig['DonationAmount'].min()), np.log10(items_orig['DonationAmount'].max()), num=13 + 1), histtype='step')
plt.gca().set_xscale('log')
plt.xlabel('Donated Amount')
plt.ylabel('#Occurrence')
plt.tight_layout()
plt.savefig('DonationAmount - Distribution of the donated amount on a logarithmic scale (for donors with at least 2 donations, excluding duplicates and low donations).pdf')
plt.close()
```

* Plot the distribution of ratings

```python
plt.figure()
plt.hist(items['DonationAmount'], histtype='step', bins=5)
plt.xlabel('Rating')
plt.ylabel('#Occurrence')
plt.tight_layout()
plt.savefig('DonationAmount - Distribution of Ratings for logarithmic bins and excluded outliers.pdf')
plt.close()
```

* Visualizing the RMSE

```python
plt.figure()
plt.grid(b=False, axis='x')

plt.errorbar(np.arange(7), [3.26, 1.05, 1.56, 3.13, 3.26, 3.26, 3.26], xerr=0.45, markersize=0., ls='none', label='Test')
plt.errorbar(np.arange(4, 7), [3.21, 3.18, 3.19], xerr=0.45, markersize=0., ls='none', label='Train')

plt.xticks(np.arange(7), ['zero', 'mean', 'random', 'SKLearn-KNN', 'SKLearn-NMF', 'SKLearn-SVD', 'SciPy-SVD'])
plt.gcf().autofmt_xdate()

plt.xlabel('Algorithm')
plt.ylabel('RMSE')
plt.tight_layout()
plt.legend(loc=4)

plt.savefig('Collaborative Filters - RMSE for DIY algorithms and some baselines.pdf')
plt.close()
```
