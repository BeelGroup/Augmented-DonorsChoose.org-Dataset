import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import surprise as spl
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.neighbors import NearestNeighbors
from surprise.model_selection import KFold as spl_KFold
from surprise.reader import Reader as spl_Reader


def rmse(estimate, truth):
    """Calculate the Root Mean Square Error for Recommender System tasks.

    Compare the non-zero entries of the actual data to the ones from the estimated data and calculate the error.

    Args:
        estimate: An estimate respectively a prediction for the data.
        truth: The actual result hence the true value of the data.

    Returns:
        rmse: Distance measure between the truth and the estimate.

    Raises:
        ValueError: If the shapes of the estimate and the truth are not the same after selecting the non-zero entries of the actual data.
    """
    if estimate.ndim != 1:
        estimate = np.asarray(estimate[truth.nonzero()]).flatten()
    truth = np.asarray(truth[truth.nonzero()]).flatten()

    if estimate.shape != truth.shape:
        raise ValueError('estimate and truth are of different shapes after selecting the non-zero entries of the actual data')

    return np.sqrt(mean_squared_error(estimate, truth))


def mae(estimate, truth):
    """Calculate the Mean Absolute Error for Recommender System tasks.

    Compare the non-zero entries of the actual data to the ones from the estimated data and calculate the error.

    Args:
        estimate: An estimate respectively a prediction for the data.
        truth: The actual result hence the true value of the data.

    Returns:
        mae: Distance measure between the truth and the estimate.

    Raises:
        ValueError: If the shapes of the estimate and the truth are not the same after selecting the non-zero entries of the actual data.
    """
    if estimate.ndim != 1:
        estimate = np.asarray(estimate[truth.nonzero()]).flatten()
    truth = np.asarray(truth[truth.nonzero()]).flatten()

    if estimate.shape != truth.shape:
        raise ValueError('estimate and truth are of different shapes after selecting the non-zero entries of the actual data')

    return mean_absolute_error(estimate, truth)


class CollaborativeFilters(object):
    """Various collaborative filtering techniques designed to recommend items to a user.

    Selection of recommender systems using collaborative filtering techniques build upon Scikit-Learn and SciPy plus SciKit-Surprise a.k.a. Surpriselib.

    Args:
        items: Table of itemized transactions with a column for user-IDs and item-IDs plus the respective rating.
        item_columns: Tuple of the form (user_column_name, item_column_name, user_item_pair_rating).
        rating_scores: Range within which the user_item_pair_rating values are allowed to vary within.
        algorithms_args: Arguments in the form of a dictionary for each algorithm to be used.
        accuracy_methods: Dictionary of accuracy methods.
        log_level: Level at which to print messages to the console.

    Attributes:
        items: Table containing the original input plus predictions for algorithms from the set in algorithms_name.
        algorithms_name: Name of all algorithms which were fitted to the data and for which subsequently columns were added to the column.
        u: Name of the column containing the user-IDs.
        i: Name of the column containing the item-IDs.
        r: Name of the column containing the ratings.
        algorithms_args: Arguments in the form of a dictionary for each algorithm to be used.
        accuracy_methods: Dictionary of accuracy methods.
        sparse_rating_matrix: Matrix in SciPy's Compressed Sparse Row form containing a pivotted view of the itemized transactions.
        sparsity: Sparsity of the pivotted itemized transaction table.
        user_ids: Unique user-IDs useful for translating between the sparse_rating_matrix and the itemized table.
        item_ids: Unique item-IDs useful for translating between the sparse_rating_matrix and the itemized table.
    """

    def __init__(self, items, item_columns, rating_scores, algorithms_args=None, accuracy_methods=None, log_level=None):
        self.items = items  # Transaction information in an itemized table
        self.u, self.i, self.r = item_columns  # User, Item, Rating/Transaction-Strength
        self.algorithms_args = defaultdict() if algorithms_args is None else algorithms_args
        self.accuracy_methods = {'RMSE': rmse, 'MAE': mae} if accuracy_methods is None else accuracy_methods
        # Keep track of the columns added to the dataframe
        self.algorithms_name = set()

        log_level = 10 if log_level is None else log_level
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(log_level)

        # Create a sparse matrix for further analysis
        self.user_ids = self.items[self.u].unique()
        self.item_ids = self.items[self.i].unique()
        ratings = self.items[self.r].values
        row = self.items[self.u].astype(pd.api.types.CategoricalDtype(categories=self.user_ids)).cat.codes
        col = self.items[self.i].astype(pd.api.types.CategoricalDtype(categories=self.item_ids)).cat.codes
        # Utilize a Compressed Sparse Row matrix as most users merely donate once or twice
        self.sparse_rating_matrix = csr_matrix((ratings, (row, col)), shape=(self.user_ids.shape[0], self.item_ids.shape[0]))

        self.sparsity = 1.0 - self.sparse_rating_matrix.nonzero()[0].shape[0] / np.dot(*self.sparse_rating_matrix.shape)
        self._logger.debug('rating matrix is {:.4%} sparse'.format(self.sparsity))

        for baseline_name, baseline_val in [('zero', np.zeros(self.sparse_rating_matrix.data.shape[0])), ('mean', np.full(self.sparse_rating_matrix.data.shape[0], self.sparse_rating_matrix.data.mean())), ('random', np.random.uniform(low=min(rating_scores), high=max(rating_scores), size=self.sparse_rating_matrix.data.shape[0]))]:
            log_line = '{:<8s} ::'.format(baseline_name)
            for acc_name, acc in sorted(self.accuracy_methods.items(), key=lambda x: x[0]):  # Predictable algorithm order for pretty printing
                overall_acc = acc(baseline_val, self.sparse_rating_matrix)
                log_line += ' | Overall-{0:s}: {overall_acc:>7.2f}'.format(acc_name, overall_acc=overall_acc)

            self._logger.debug(log_line)

        # Read data into scikit-surprise respectively surpriselib
        spl_reader = spl_Reader(line_format='user item rating', rating_scale=(int(min(rating_scores)), int(max(rating_scores))))
        self.spl_items = spl.Dataset.load_from_df(self.items[[self.u, self.i, self.r]], spl_reader)

    def fit_all(self, n_folds=5):
        """Fit everything and return the instance of the class.

        Convenient helper function which performs every available collaborative filtering technique and amends the itemized transaction table accordingly.

        Args:
            n_folds: Number of folds to perform in cross-validation.
        """
        sci_algorithms = {}
        sci_algorithms['SciPy-SVD'] = self.SciPySVD(**self.algorithms_args['SciPy-SVD'])
        sci_algorithms['SKLearn-SVD'] = self.SKLearnSVD(**self.algorithms_args['SKLearn-SVD'])
        sci_algorithms['SKLearn-KNN'] = self.SKLearnKNN(**self.algorithms_args['SKLearn-KNN'])
        sci_algorithms['SKLearn-NMF'] = self.SKLearnNMF(**self.algorithms_args['SKLearn-NMF'])
        self.algorithms_name.update(sci_algorithms.keys())
        # Initialize a dictionary with an entry for each algorithm which shall store accuracy values for every selected accuracy method
        algorithms_error = {}
        for alg_name in sci_algorithms.keys():
            # Tuple of training error and test error for each algorithm
            algorithms_error[alg_name] = {acc_name: np.array([0., 0.]) for acc_name in self.accuracy_methods.keys()}

        kf = KFold(n_splits=n_folds, shuffle=True)

        i = 0
        # The ordering the indices of the matrix and the user_ids, item_ids of the frame must match in order to merge the prediction back into the table
        # By default the created sparse matrix has sorted indices. However, act with caution when working with subsets of the matrix!
        for train_idx, test_idx in kf.split(self.sparse_rating_matrix):
            i += 1

            for alg_name, alg in sorted(sci_algorithms.items(), key=lambda x: x[0]):  # Predictable algorithm order for reproducibility
                log_line = '{:<15s} (fold {:>d}/{:<d}) ::'.format(alg_name, i, n_folds)
                train_predictions = alg.fit_transform(self.sparse_rating_matrix[train_idx].sorted_indices())
                test_predictions = alg.estimate(self.sparse_rating_matrix[test_idx].sorted_indices())

                # Extract the test predictions from the matrix by selecting the proper indices from the test_idx and the matrix
                user_merge_idx, item_merge_idx = self.sparse_rating_matrix.nonzero()
                merge_users = np.isin(user_merge_idx, test_idx)  # The test_idx is a subset of the rows of the matrix
                user_merge_idx, item_merge_idx = user_merge_idx[merge_users], item_merge_idx[merge_users]
                items_prediction = pd.DataFrame({'Prediction' + alg_name: np.asarray(test_predictions[self.sparse_rating_matrix[test_idx].sorted_indices().nonzero()]).flatten(), self.u: self.user_ids[user_merge_idx], self.i: self.item_ids[item_merge_idx]})
                self.items = pd.merge(self.items, items_prediction, on=[self.u, self.i], how='left', suffixes=('_x', '_y'), sort=False)
                # Add predictions (NaN is treated as zero here) and remove separate results if necessary
                if 'Prediction' + alg_name + '_x' in self.items.columns and 'Prediction' + alg_name + '_y' in self.items.columns:
                    self.items['Prediction' + alg_name] = self.items[['Prediction' + alg_name + '_x', 'Prediction' + alg_name + '_y']].sum(axis=1)
                    self.items = self.items.drop(['Prediction' + alg_name + '_x', 'Prediction' + alg_name + '_y'], axis=1)

                for acc_name, acc in sorted(self.accuracy_methods.items(), key=lambda x: x[0]):  # Predictable algorithm order for pretty printing
                    train_acc, test_acc = acc(train_predictions, self.sparse_rating_matrix[train_idx].sorted_indices()), acc(test_predictions, self.sparse_rating_matrix[test_idx].sorted_indices())
                    algorithms_error[alg_name][acc_name] += np.array([train_acc, test_acc]) / n_folds
                    log_line += ' | Training-{0:s}: {train_acc:>7.2f}, Test-{0:s}: {test_acc:>7.2f}'.format(acc_name, train_acc=train_acc, test_acc=test_acc)

                self._logger.debug(log_line)

        for alg_name, acc_methods in sorted(algorithms_error.items(), key=lambda x: x[0]):
            log_line = '{:<15s} (average) ::'.format(alg_name)
            for acc_name, acc_value in sorted(acc_methods.items(), key=lambda x: x[0]):
                log_line += ' | Training-{0:s}: {train_acc:>7.2f}, Test-{0:s}: {test_acc:>7.2f}'.format(acc_name, train_acc=acc_value[0], test_acc=acc_value[1])

            self._logger.debug(log_line)

        spl_algorithms = {}
        spl_algorithms['SPL-SVD'] = spl.SVD(**self.algorithms_args['SPL-SVD'])
        spl_algorithms['SPL-SVDpp'] = spl.SVDpp(**self.algorithms_args['SPL-SVDpp'])
        spl_algorithms['SPL-NMF'] = spl.NMF(**self.algorithms_args['SPL-NMF'])
        spl_algorithms['SPL-KNNWithMeans'] = spl.KNNWithMeans(**self.algorithms_args['SPL-KNNWithMeans'])
        spl_algorithms['SPL-KNNBasic'] = spl.KNNBasic(**self.algorithms_args['SPL-KNNBasic'])
        spl_algorithms['SPL-KNNWithZScore'] = spl.KNNWithZScore(**self.algorithms_args['SPL-KNNWithZScore'])
        spl_algorithms['SPL-KNNBaseline'] = spl.KNNBaseline(**self.algorithms_args['SPL-KNNBaseline'])
        spl_algorithms['SPL-NormalPredictor'] = spl.NormalPredictor(**self.algorithms_args['SPL-NormalPredictor'])
        spl_algorithms['SPL-CoClustering'] = spl.CoClustering(**self.algorithms_args['SPL-CoClustering'])
        spl_algorithms['SPL-SlopeOne'] = spl.SlopeOne(**self.algorithms_args['SPL-SlopeOne'])
        self.algorithms_name.update(spl_algorithms.keys())

        spl_kf = spl_KFold(n_splits=n_folds, shuffle=True)
        for spl_train, spl_test in spl_kf.split(self.spl_items):
            for alg_name, alg in sorted(spl_algorithms.items(), key=lambda x: x[0]):
                alg.fit(spl_train)
                # Test returns an object of type surprise.prediction_algorithms.predictions.Prediction
                predictions = pd.DataFrame(alg.test(spl_test), columns=[self.u, self.i, 'TrueRating', 'Prediction' + alg_name, 'RatingDetails'])
                # Merge the predicted rating into the dataframe
                self.items = pd.merge(self.items, predictions[[self.u, self.i, 'Prediction' + alg_name]], on=[self.u, self.i], how='left', suffixes=('_x', '_y'), sort=False)
                if 'Prediction' + alg_name + '_x' in self.items.columns and 'Prediction' + alg_name + '_y' in self.items.columns:
                    self.items['Prediction' + alg_name] = self.items[['Prediction' + alg_name + '_x', 'Prediction' + alg_name + '_y']].sum(axis=1)
                    self.items = self.items.drop(['Prediction' + alg_name + '_x', 'Prediction' + alg_name + '_y'], axis=1)

        # Overall accuracy
        for alg_name in sorted(self.algorithms_name):
            log_line = '{:<20s} (overall) ::'.format(alg_name)
            for acc_name, acc in sorted(self.accuracy_methods.items(), key=lambda x: x[0]):
                overall_acc = acc(self.items['Prediction' + alg_name].values, np.asarray(self.items[self.r]))
                log_line += ' | Overall-{0:s}: {overall_acc:>7.2f}'.format(acc_name, overall_acc=overall_acc)

            self._logger.info(log_line)

        return self

    class SciPySVD(object):
        """SciPy's Singular Value Decomposition adapted for Recommender System tasks.

        Decompose a matrix using SciPy and store the decomposition in the instance of the class to allow further evaluation.

        Methods:
            fit_transform(X[, y]): Fit model to training data and return the estimate for the input - with target y being ignored.
            estimate(X): Predict target using fitted model.
        """

        def __init__(self, n_components=None, **kwargs):
            """Initialize internal attributes of the class.

            Args:
                n_components: The number of singular values which to select upon decomposition. Defaults to 100.
                kwargs: Dictionary of additional arguments which are passed to the underlying algorithm.
            """
            self.n_components = 100 if n_components is None else n_components
            self.kwargs = {} if kwargs is None else kwargs

            self.u, self.s, self.vt = None, None, None

        def fit_transform(self, train_set, y=None):
            """Perform the decomposition, store the resulting matrices for later estimations and return the final estimate for the input.

            Args:
                train_set: Dataset used for training.
                y: Ignored optional target variable; present only for compatibility reasons.

            Returns:
                self: The instance of the fitted model.
            """
            self.u, self.s, self.vt = svds(train_set, k=self.n_components, **self.kwargs)
            self.s = np.diag(self.s)

            return self.u.dot(self.s).dot(self.vt)

        def estimate(self, test_set):
            """Return an estimate of the given input data using the fit parameters of the model.

            Args:
                test_set: Dataset used for training.

            Returns:
                Estimation of the set using the fitted model for transformation.
            """
            # Matrix 'u' corresponds to the user features and shall now be replaced by the test users
            # Hence, to get our new 'u' from the input, one inverts 'vt' and 's'. Bear in mind, 's' is a real quadratic diagonal matrix while 'vt' is a non-quadratic unitary matrix.
            return test_set.dot(self.vt.T).dot(self.vt)

    class SKLearnSVD(object):
        """SciKit-Learn's Singular Value Decomposition adapted for Recommender System tasks.

        Decompose a matrix using SciKit-Learn and store the decomposition in the instance of the class to allow further evaluation.

        Methods:
            fit_transform(X[, y]): Fit model to training data and return the estimate for the input - with target y being ignored.
            estimate(X): Predict target using fitted model.
        """

        def __init__(self, n_components=None, **kwargs):
            """Initialize internal attributes of the class.

            Args:
                n_components: The number of singular values which to select upon decomposition. Defaults to 100.
                kwargs: Dictionary of additional arguments which are passed to the underlying algorithm.
            """
            self.n_components = 100 if n_components is None else n_components
            self.kwargs = {} if kwargs is None else kwargs

            self.svd = TruncatedSVD(n_components=n_components, **kwargs)

        def fit_transform(self, train_set, y=None):
            """Perform the decomposition, store the resulting matrices for later estimations and return the final estimate for the input.

            Args:
                train_set: Dataset used for training.
                y: Ignored optional target variable; present only for compatibility reasons.

            Returns:
                self: The instance of the fitted model.
            """
            return self.svd.inverse_transform(self.svd.fit_transform(train_set))

        def estimate(self, test_set):
            """Return an estimate of the given input data using the fit parameters of the model.

            Args:
                test_set: Dataset used for training.

            Returns:
                Estimation of the set using the fitted model for transformation.
            """
            return self.svd.inverse_transform(self.svd.transform(test_set))

    class SKLearnNMF(object):
        """SciKit-Learn's Non-Negative Matrix Factorization adapted for Recommender System tasks.

        Decompose a matrix using SciKit-Learn and store the decomposition in the instance of the class to allow further evaluation.

        Methods:
            fit_transform(X[, y]): Fit model to training data and return the estimate for the input - with target y being ignored.
            estimate(X): Predict target using fitted model.
        """

        def __init__(self, n_components=None, **kwargs):
            """Initialize internal attributes of the class.

            Args:
                n_components: The number of components which to select upon decomposition. Defaults to 50.
                kwargs: Dictionary of additional arguments which are passed to the underlying algorithm.
            """
            self.n_components = 50 if n_components is None else n_components
            self.kwargs = {} if kwargs is None else kwargs

            self.nmf = NMF(n_components=n_components, **kwargs)

        def fit_transform(self, train_set, y=None):
            """Perform the decomposition, store the resulting matrices for later estimations and return the final estimate for the input.

            Args:
                train_set: Dataset used for training.
                y: Ignored optional target variable; present only for compatibility reasons.

            Returns:
                self: The instance of the fitted model.
            """
            return self.nmf.inverse_transform(self.nmf.fit_transform(train_set))

        def estimate(self, test_set):
            """Return an estimate of the given input data using the fit parameters of the model.

            Args:
                test_set: Dataset used for training.

            Returns:
                Estimation of the set using the fitted model for transformation.
            """
            return self.nmf.inverse_transform(self.nmf.transform(test_set))

    class SKLearnKNN(object):
        """SciKit-Learn's k-Nearest-Neighbor clustering algorithm adapted for Recommender System tasks.

        Find the nearest neighbors in a set and store the clusters in the instance of the class to allow further evaluation.

        Methods:
            fit_transform(X[, y]): Fit model to training data and return the estimate for the input - with target y being ignored.
            estimate(X): Predict target using fitted model.
        """

        def __init__(self, n_neighbors=None, **kwargs):
            """Initialize internal attributes of the class.

            Args:
                n_neighbors: The number of neighbors to use for nearest neighbor queries. Defaults to 40.
                kwargs: Dictionary of additional arguments which are passed to the underlying algorithm. Defaults to the cosine as distance metric and the brute algorithm.
            """
            self.n_neighbors = 40 if n_neighbors is None else n_neighbors
            self.kwargs = {'distance': 'cosine', 'algorithm': 'brute'} if kwargs is None else kwargs

            self.knn = NearestNeighbors(n_neighbors=n_neighbors, **kwargs)

        def fit_transform(self, train_set, y=None):
            """Perform the nearest neighboor clustering, store the resulting parameters for later estimations and return the final estimate for the input.

            Args:
                train_set: Dataset used for training.
                y: Ignored optional target variable; present only for compatibility reasons.

            Returns:
                self: The instance of the fitted model.
            """
            self.knn.fit(train_set)
            self.population_matrix = train_set.copy()  # K-NN queries depend on the training dataset to be present
            indices = self.knn.kneighbors(train_set, n_neighbors=1, return_distance=False).flatten()
            return self.population_matrix[indices]

        def estimate(self, test_set):
            """Return an estimate of the given input data using the fit parameters of the model.

            Args:
                test_set: Dataset used for training.

            Returns:
                Estimation of the set using the fitted model for transformation.
            """
            indices = self.knn.kneighbors(test_set, n_neighbors=1, return_distance=False).flatten()
            return self.population_matrix[indices]


class ContentFilers(object):
    """Various content-based filtering techniques designed to recommend items to a user.

    Selection of recommender systems using content-based filtering techniques build upon Scikit-Learn and SciPy.

    Args:
        items: Table of itemized transactions with a column for user-IDs and item-IDs plus the respective rating.
        column_names: Tuple of the form (user_column_name, item_column_name, user_item_pair_rating).
        content_items: Table of itemized item information with a column mapping the item-IDs to the items table.
        content_column_names: Tuple containing the columns which are to be concatenated and used as input features.
        algorithms_args: Arguments in the form of a dictionary for each algorithm to be used.
        log_level: Level at which to print messages to the console.

    Attributes:
        items: Table containing the original input plus predictions for algorithms from the set algorithms_name.
        algorithms_name: Name of all algorithms which were fitted to the data and for which subsequently columns were added to the column.
        u: Name of the column containing the user-IDs.
        i: Name of the column containing the item-IDs.
        r: Name of the column containing the ratings.
        content_items: Table containing the original input plus predictions for algorithms from the set in algorithms_name.
        t: Name of the column containing the input for the algorithms.
        tfidf_matrix: Matrix containing the TF-IDF features.
        algorithms_args: Arguments in the form of a dictionary for each algorithm to be used.
    """

    def __init__(self, items, item_columns, content_items, content_item_columns, algorithms_args=None, log_level=None):
        self.items = items  # Transaction information in an itemized table
        self.u, self.i, self.r = item_columns  # User, Item, Rating/Transaction-Strength

        self.t = self.__class__.__name__ + 'Input'

        self.content_items = content_items  # Item information in an itemized table
        self.algorithms_args = defaultdict() if algorithms_args is None else algorithms_args
        # Keep track of the columns added to the dataframe
        self.algorithms_name = set()

        self.tfidf_matrix = np.nan

        log_level = 10 if log_level is None else log_level
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(log_level)

        # Select only those items which were previously used for the collaborative filtering approach
        self.content_items = self.content_items[self.content_items[self.i].isin(self.items[self.i])].fillna('')
        self.content_items[self.t] = ''
        for column in content_item_columns:
            self.content_items[self.t] += ' ' + self.content_items[column]
        self.content_items = self.content_items[[self.i, self.t]]

    def fit_all(self, n_random_non_interacted_items=100):
        """Fit everything and return the instance of the class.

        Convenient helper function which performs every available content-based filtering technique and amends the itemized transaction table accordingly.

        Args:
            n_random_non_interacted_items: Number of random non-interacted items which shall be used for calculating the accuracy metrics.
        """
        content_algorithms = {}
        content_algorithms['SKLearn-TfidfVectorizer'] = TfidfVectorizer(**self.algorithms_args['SKLearn-TfidfVectorizer'])
        self.algorithms_name.update(content_algorithms.keys())
        alg_name, alg = 'SKLearn-TfidfVectorizer', content_algorithms['SKLearn-TfidfVectorizer']

        content_item_ids = self.content_items[self.i].values
        self.tfidf_matrix = alg.fit_transform(self.content_items[self.t])
        # One may pretty print actual words/n-grams instead of positions in arrays using `tfidf_feature_names = alg.get_feature_names()`

        self.items['RecallAtPosition' + alg_name] = np.nan
        loo = LeaveOneOut()
        for user in self.items[self.u].unique():  # This loop is computationally expensive
            user_items = self.items[self.items[self.u] == user]
            user_item_idx = np.isin(content_item_ids, user_items[self.i])
            user_features = self.tfidf_matrix[user_item_idx]

            # Single out one interacted item to use for testing using integer indices
            for train_user_features_idx, test_user_features_idx in loo.split(user_features):
                test_user_feature = user_items[self.i].iloc[test_user_features_idx].values

                # Multiply each item's feature with the user's rating of the item and divide the result by the sum of all ratings made by the user
                content_train_users_profiles = user_features[train_user_features_idx].transpose().dot(user_items[self.r].iloc[train_user_features_idx].values)
                content_train_users_profiles = content_train_users_profiles / user_items[self.r].iloc[train_user_features_idx].sum()

                # Create an array of items which the user has not interacted with yet plus one with which an interaction took place
                non_rated_user_items_ids = np.setdiff1d(content_item_ids, user_items[self.i])
                top_test_choice = np.append(np.random.choice(np.array(non_rated_user_items_ids), n_random_non_interacted_items), test_user_feature)
                top_test_idx = np.isin(content_item_ids, top_test_choice)

                # Calculate the similarity and retrieve the position of the test id within the recommended set
                cosine_similarities = cosine_similarity(content_train_users_profiles.reshape(1, -1), self.tfidf_matrix[top_test_idx]).flatten()
                sorted_similarities_indices = (-1 * cosine_similarities).argsort()
                sorted_item_ids = content_item_ids[top_test_idx][sorted_similarities_indices]
                pos = np.where(sorted_item_ids == test_user_feature[0])[0][0]

                self.items.at[(self.items[self.u] == user) & (self.items[self.i] == test_user_feature[0]), 'RecallAtPosition' + alg_name] = pos

        return self
