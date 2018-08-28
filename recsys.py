import logging

import numpy as np
import pandas as pd
import surprise as spl
from gensim.models import FastText
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
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

    Selection of recommender systems using collaborative filtering techniques build upon Scikit-Learn and SciPy.

    Args:
        items: Table of itemized transactions with a column for user-IDs and item-IDs plus the respective rating.
        item_columns: Tuple of the form (user_column_name, item_column_name, user_item_pair_rating).
        rating_scores: Range within which the user_item_pair_rating values are allowed to vary within.
        algorithms_args: Arguments in the form of a dictionary for each algorithm to be used. Possible keys are {'SciPy-SVD', 'SKLearn-SVD', 'SKLearn-KNN', 'SKLearn-NMF'}, other keys will be silently ignored.
        accuracy_methods: Set or array of names for accuracy methods which to use for evaluation. Accepts a subset of {'RMSE', 'MAE', 'RecallAtPosition'}.
        log_level: Level at which to print messages to the console.

    Attributes:
        items: Table containing the original input plus predictions for algorithms from the set in algorithms_name.
        algorithms_name: Name of all algorithms which were fitted to the data and for which subsequently columns were added to the column.
        u: Name of the column containing the user-IDs.
        i: Name of the column containing the item-IDs.
        r: Name of the column containing the ratings.
        algorithms_avail: Dictionary of available algorithms with the location of the function as the value.
        algorithms_args: Arguments in the form of a dictionary for each algorithm to be used.
        accuracy_methods: Set or array of names for accuracy methods which to use for evaluation.
        sparse_rating_matrix: Matrix in SciPy's Compressed Sparse Row form containing a pivotted view of the itemized transactions.
        sparsity: Sparsity of the pivotted itemized transaction table.
        user_ids: Unique user-IDs useful for translating between the sparse_rating_matrix and the itemized table.
        item_ids: Unique item-IDs useful for translating between the sparse_rating_matrix and the itemized table.
    """

    def __init__(self, items, item_columns, rating_scores, algorithms_args=None, accuracy_methods=None, log_level=None):
        self.items = items  # Transaction information in an itemized table
        self.u, self.i, self.r = item_columns  # User, Item, Rating/Transaction-Strength

        self.algorithms_avail = {'SciPy-SVD': self.SciPySVD, 'SKLearn-SVD': self.SKLearnSVD, 'SKLearn-KNN': self.SKLearnKNN, 'SKLearn-NMF': self.SKLearnNMF}

        self.algorithms_args = dict.fromkeys(self.algorithms_avail.keys(), {}) if algorithms_args is None else algorithms_args
        self.accuracy_methods = {'RMSE', 'MAE', 'RecallAtPosition'} if accuracy_methods is None else accuracy_methods
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
            for acc_name in sorted(self.accuracy_methods):  # Predictable algorithm order for pretty printing
                if acc_name == 'RMSE':
                    overall_acc = rmse(baseline_val, self.sparse_rating_matrix)
                elif acc_name == 'MAE':
                    overall_acc = mae(baseline_val, self.sparse_rating_matrix)
                elif acc_name == 'RecallAtPosition':
                    continue
                else:
                    raise ValueError('Expected a valid name for an accuracy method, got "{}".'.format(acc_name))

                log_line += ' | Overall-{0:s}: {overall_acc:>7.2f}'.format(acc_name, overall_acc=overall_acc)

            self._logger.debug(log_line)

    def fit_all(self, n_folds=5, n_random_non_interacted_items=100):
        """Fit everything and return the instance of the class.

        Convenient helper function which performs every available collaborative filtering technique and amends the itemized transaction table accordingly.

        Args:
            n_folds: Number of folds to perform in cross-validation.
        """
        sci_algorithms = {}
        for alg_name in np.intersect1d(list(self.algorithms_avail.keys()), list(self.algorithms_args.keys())):
            sci_algorithms[alg_name] = self.algorithms_avail[alg_name](**self.algorithms_args[alg_name])

        if not sci_algorithms:
            self._logger.warning('{} method invoked without having specified any tasks; Returning...'.format(self.__class__.__name__))
            return self

        self.algorithms_name.update(sci_algorithms.keys())
        # Initialize a dictionary with an entry for each algorithm which shall store accuracy values for every selected accuracy method
        algorithms_error = {}
        for alg_name in sci_algorithms.keys():
            # Tuple of training error and test error for each algorithm
            algorithms_error[alg_name] = {acc_name: np.array([0., 0.]) for acc_name in self.accuracy_methods}

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

                for acc_name in sorted(self.accuracy_methods):  # Predictable algorithm order for pretty printing
                    if acc_name == 'RMSE':
                        # This could in theory be done once at the end but let's keep it here eitherway to make each cross-validation step fully independant
                        if 'SquareError' + alg_name not in self.items:
                            self.items['SquareError' + alg_name] = np.nan

                        test_item_idx = self.items['SquareError' + alg_name].isna() & self.items['Prediction' + alg_name].notna()
                        self.items.at[test_item_idx, 'SquareError' + alg_name] =  np.square(self.items[test_item_idx][self.r] - self.items[test_item_idx]['Prediction' + alg_name])

                        train_acc = rmse(train_predictions, self.sparse_rating_matrix[train_idx].sorted_indices())
                        test_acc = rmse(test_predictions, self.sparse_rating_matrix[test_idx].sorted_indices())
                    elif acc_name == 'MAE':
                        if 'AbsoluteError' + alg_name not in self.items:
                            self.items['AbsoluteError' + alg_name] = np.nan

                        test_item_idx = self.items['AbsoluteError' + alg_name].isna() & self.items['Prediction' + alg_name].notna()
                        self.items.at[test_item_idx, 'AbsoluteError' + alg_name] =  np.square(self.items[test_item_idx][self.r] - self.items[test_item_idx]['Prediction' + alg_name])

                        train_acc = mae(train_predictions, self.sparse_rating_matrix[train_idx].sorted_indices())
                        test_acc = mae(test_predictions, self.sparse_rating_matrix[test_idx].sorted_indices())
                    elif acc_name == 'RecallAtPosition':
                        loo = LeaveOneOut()
                        for user, user_row in zip(self.user_ids[test_idx], self.sparse_rating_matrix[test_idx]):  # This loop is computationally expensive
                            user_nonzero_idx = user_row.nonzero()[1]
                            # Splitting the data returns indices; However, be aware that indices were already handled in user_nonzero_idx
                            for _, test_user_idx in loo.split(user_nonzero_idx):
                                user_row_train = user_row.copy()
                                user_row_train[0, user_nonzero_idx[test_user_idx[0]]] = 0
                                user_train_prediction = alg.estimate(user_row_train)
                                if type(user_train_prediction) is csr_matrix:
                                    user_train_prediction = user_train_prediction.toarray()

                                user_train_prediction = user_train_prediction.flatten()

                                non_rated_items_idx = np.setdiff1d(np.arange(user_row.shape[1]), user_nonzero_idx)
                                top_test_idx_choice = np.append(np.random.choice(non_rated_items_idx, n_random_non_interacted_items), user_nonzero_idx[test_user_idx])

                                sorted_top_prediction_idx = (-1 * user_train_prediction[top_test_idx_choice]).argsort()
                                # The true rating is the last entry in top_test_idx_choice and hence at the position of n_random_non_interacted_items
                                pos = np.where(sorted_top_prediction_idx == n_random_non_interacted_items)[0][0]

                                self.items.at[(self.items[self.u] == user) & (self.items[self.i] == self.item_ids[user_nonzero_idx[test_user_idx[0]]]), 'RecallAtPosition' + alg_name] = pos

                        train_acc = 0
                        test_acc = self.items[self.items[self.u].isin(self.user_ids[test_idx])]['RecallAtPosition' + alg_name].mean()
                    else:
                        raise ValueError('Expected a valid name for an accuracy method, got "{}".'.format(acc_name))

                    algorithms_error[alg_name][acc_name] += np.array([train_acc, test_acc]) / n_folds
                    log_line += ' | Training-{0:s}: {train_acc:>7.2f}, Test-{0:s}: {test_acc:>7.2f}'.format(acc_name, train_acc=train_acc, test_acc=test_acc)

                self._logger.debug(log_line)

        for alg_name, acc_methods in sorted(algorithms_error.items(), key=lambda x: x[0]):
            log_line = '{:<15s} (average) ::'.format(alg_name)
            for acc_name, acc_value in sorted(acc_methods.items(), key=lambda x: x[0]):
                log_line += ' | Training-{0:s}: {train_acc:>7.2f}, Test-{0:s}: {test_acc:>7.2f}'.format(acc_name, train_acc=acc_value[0], test_acc=acc_value[1])

            self._logger.debug(log_line)

        # Overall accuracy
        for alg_name in sorted(self.algorithms_name):
            log_line = '{:<20s} (overall) ::'.format(alg_name)
            for acc_name in sorted(self.accuracy_methods):
                if acc_name == 'RMSE':
                    overall_acc = rmse(self.items['Prediction' + alg_name].values, np.asarray(self.items[self.r]))
                elif acc_name == 'MAE':
                    overall_acc = mae(self.items['Prediction' + alg_name].values, np.asarray(self.items[self.r]))
                elif acc_name == 'RecallAtPosition':
                    overall_acc = self.items['RecallAtPosition' + alg_name].mean()
                else:
                    raise ValueError('Expected a valid name for an accuracy method, got "{}".'.format(acc_name))

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


class CollaborativeFiltersSpl(object):
    """Various content-based filtering techniques designed to recommend items to a user.

    Selection of recommender systems using content-based filtering techniques build upon SciKit-Surprise a.k.a Surpriselib.

    Args:
        items: Table of itemized transactions with a column for user-IDs and item-IDs plus the respective rating.
        item_columns: Tuple of the form (user_column_name, item_column_name, user_item_pair_rating).
        rating_scores: Range within which the user_item_pair_rating values are allowed to vary within.
        algorithms_args: Arguments in the form of a dictionary for each algorithm to be used. Possible keys are {'SPL-SVD', 'SPL-SVDpp', 'SPL-NMF', 'SPL-KNNWithMeans', 'SPL-KNNBasic', 'SPL-KNNWithZScore', 'SPL-KNNBaseline', 'SPL-NormalPredictor', 'SPL-CoClustering', 'SPL-SlopeOne'}, other keys will be silently ignored.
        accuracy_methods: Set or array of names for accuracy methods which to use for evaluation. Accepts a subset of {'RMSE', 'MAE', 'RecallAtPosition'} with 'RecallAtPosition' being silently ignored.
        log_level: Level at which to print messages to the console.

    Attributes:
        items: Table containing the original input plus predictions for algorithms from the set in algorithms_name.
        spl_items: Table containing the data in a format suitable for Surpriselib.
        algorithms_name: Name of all algorithms which were fitted to the data and for which subsequently columns were added to the column.
        u: Name of the column containing the user-IDs.
        i: Name of the column containing the item-IDs.
        r: Name of the column containing the ratings.
        algorithms_avail: Dictionary of available algorithms with the location of the function as the value.
        algorithms_args: Arguments in the form of a dictionary for each algorithm to be used.
        accuracy_methods: Set or array of names for accuracy methods which to use for evaluation.
    """

    def __init__(self, items, item_columns, rating_scores, algorithms_args=None, accuracy_methods=None, log_level=None):
        self.items = items  # Transaction information in an itemized table
        self.u, self.i, self.r = item_columns  # User, Item, Rating/Transaction-Strength

        self.algorithms_avail = {'SPL-SVD': spl.SVD, 'SPL-SVDpp': spl.SVDpp, 'SPL-NMF': spl.NMF, 'SPL-KNNWithMeans': spl.KNNWithMeans, 'SPL-KNNBasic': spl.KNNBasic, 'SPL-KNNWithZScore': spl.KNNWithZScore, 'SPL-KNNBaseline': spl.KNNBaseline, 'SPL-NormalPredictor': spl.NormalPredictor, 'SPL-CoClustering': spl.CoClustering, 'SPL-SlopeOne': spl.SlopeOne}

        self.algorithms_args = dict.fromkeys(self.algorithms_avail.keys(), {}) if algorithms_args is None else algorithms_args
        self.accuracy_methods = {'RMSE', 'MAE'} if accuracy_methods is None else accuracy_methods
        # Keep track of the columns added to the dataframe
        self.algorithms_name = set()

        log_level = 10 if log_level is None else log_level
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(log_level)

        # Read data into scikit-surprise respectively surpriselib
        spl_reader = spl_Reader(line_format='user item rating', rating_scale=(int(min(rating_scores)), int(max(rating_scores))))
        self.spl_items = spl.Dataset.load_from_df(self.items[[self.u, self.i, self.r]], spl_reader)

    def fit_all(self, n_folds=5):
        """Fit everything and return the instance of the class.

        Convenient helper function which performs every available content-based filtering technique and amends the itemized transaction table accordingly.

        Args:
            n_folds: Number of folds to perform in cross-validation.
        """
        spl_algorithms = {}
        for alg_name in np.intersect1d(list(self.algorithms_avail.keys()), list(self.algorithms_args.keys())):
            spl_algorithms[alg_name] = self.algorithms_avail[alg_name](**self.algorithms_args[alg_name])

        if not spl_algorithms:
            self._logger.warning('{} method invoked without having specified any tasks; Returning...'.format(self.__class__.__name__))
            return self

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

                for acc_name in sorted(self.accuracy_methods):  # Predictable algorithm order for pretty printing
                    if acc_name == 'RMSE':
                        # This could in theory be done once at the end but let's keep it here eitherway to make each cross-validation step fully independant
                        if 'SquareError' + alg_name not in self.items:
                            self.items['SquareError' + alg_name] = np.nan

                        test_item_idx = self.items['SquareError' + alg_name].isna() & self.items['Prediction' + alg_name].notna()
                        self.items.at[test_item_idx, 'SquareError' + alg_name] =  np.square(self.items[test_item_idx][self.r] - self.items[test_item_idx]['Prediction' + alg_name])
                    elif acc_name == 'MAE':
                        if 'AbsoluteError' + alg_name not in self.items:
                            self.items['AbsoluteError' + alg_name] = np.nan

                        test_item_idx = self.items['AbsoluteError' + alg_name].isna() & self.items['Prediction' + alg_name].notna()
                        self.items.at[test_item_idx, 'AbsoluteError' + alg_name] =  np.square(self.items[test_item_idx][self.r] - self.items[test_item_idx]['Prediction' + alg_name])
                    elif acc_name == 'RecallAtPosition':
                        continue
                    else:
                        raise ValueError('Expected a valid name for an accuracy method, got "{}".'.format(acc_name))

        # Overall accuracy
        for alg_name in sorted(self.algorithms_name):
            log_line = '{:<20s} (overall) ::'.format(alg_name)
            for acc_name in sorted(self.accuracy_methods):  # Predictable algorithm order for pretty printing
                if acc_name == 'RMSE':
                    overall_acc = rmse(self.items['Prediction' + alg_name].values, np.asarray(self.items[self.r]))
                elif acc_name == 'MAE':
                    overall_acc = mae(self.items['Prediction' + alg_name].values, np.asarray(self.items[self.r]))
                elif acc_name == 'RecallAtPosition':
                    continue
                else:
                    raise ValueError('Expected a valid name for an accuracy method, got "{}".'.format(acc_name))

                log_line += ' | Overall-{0:s}: {overall_acc:>7.2f}'.format(acc_name, overall_acc=overall_acc)

            self._logger.info(log_line)

        return self


class ContentFilers(object):
    """Various content-based filtering techniques designed to recommend items to a user.

    Selection of recommender systems using content-based filtering techniques build upon Scikit-Learn and SciPy.

    Args:
        items: Table of itemized transactions with a column for user-IDs and item-IDs plus the respective rating.
        column_names: Tuple of the form (user_column_name, item_column_name, user_item_pair_rating).
        content_items: Table of itemized item information with a column mapping the item-IDs to the items table.
        content_column_names: Tuple containing the columns which are to be concatenated and used as input features.
        algorithms_args: Arguments in the form of a dictionary for each algorithm to be used. Possible keys are {'SKLearn-TfidfVectorizer'}, other keys will be silently ignored.
        accuracy_methods: Set or array of names for accuracy methods which to use for evaluation. Accepts a subset of {'RMSE', 'MAE', 'RecallAtPosition'} with 'RMSE' and 'MAE' being silently ignored.
        log_level: Level at which to print messages to the console.

    Attributes:
        items: Table containing the original input plus predictions for algorithms from the set algorithms_name.
        algorithms_name: Name of all algorithms which were fitted to the data and for which subsequently columns were added to the column.
        u: Name of the column containing the user-IDs.
        i: Name of the column containing the item-IDs.
        r: Name of the column containing the ratings.
        content_items: Table containing the original input plus predictions for algorithms from the set in algorithms_name.
        t: Name of the column containing the input for the algorithms.
        content_matrix: Matrix containing the TF-IDF features.
        algorithms_avail: Dictionary of available algorithms with the location of the function as the value.
        algorithms_args: Arguments in the form of a dictionary for each algorithm to be used.
        accuracy_methods: Set or array of names for accuracy methods which to use for evaluation. Accepts a subset of {'RMSE', 'MAE', 'RecallAtPosition'}.
    """

    def __init__(self, items, item_columns, content_items, content_item_columns, algorithms_args=None, accuracy_methods=None, log_level=None):
        self.items = items  # Transaction information in an itemized table
        self.u, self.i, self.r = item_columns  # User, Item, Rating/Transaction-Strength

        self.t = self.__class__.__name__ + 'Input'

        self.content_items = content_items  # Item information in an itemized table
        self.accuracy_methods = {'RecallAtPosition'} if accuracy_methods is None else accuracy_methods

        self.algorithms_avail = {'SKLearn-TfidfVectorizer': TfidfVectorizer, 'Gensim-FastText': self.GensimFastText}

        self.algorithms_args = dict.fromkeys(self.algorithms_avail.keys(), {}) if algorithms_args is None else algorithms_args
        # Keep track of the columns added to the dataframe
        self.algorithms_name = set()

        self.content_matrix = np.nan

        log_level = 10 if log_level is None else log_level
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(log_level)

        # Select only those items which were previously used for the collaborative filtering approach
        self.content_items = self.content_items[self.content_items[self.i].isin(self.items[self.i])].fillna('')
        self.content_items[self.t] = ''
        for column in content_item_columns:
            self.content_items[self.t] += ' ' + self.content_items[column]
        self.content_items = self.content_items[[self.i, self.t]]

        # Remove characters which may prohibit a proper content matching
        self.content_items[self.t] = self.content_items[self.t].str.replace(r"[\"\'\\n\\t\d]", '', regex=True)

    def fit_all(self, n_random_non_interacted_items=100):
        """Fit everything and return the instance of the class.

        Convenient helper function which performs every available content-based filtering technique and amends the itemized transaction table accordingly.

        Args:
            n_random_non_interacted_items: Number of random non-interacted items which shall be used for calculating the accuracy metrics.
        """
        content_algorithms = {}
        for alg_name in np.intersect1d(list(self.algorithms_avail.keys()), list(self.algorithms_args.keys())):
            content_algorithms[alg_name] = self.algorithms_avail[alg_name](**self.algorithms_args[alg_name])

        if not content_algorithms:
            self._logger.warning('{} method invoked without having specified any tasks; Returning...'.format(self.__class__.__name__))
            return self

        self.algorithms_name.update(content_algorithms.keys())

        for alg_name, alg in sorted(content_algorithms.items(), key=lambda x: x[0]):
            content_item_ids = self.content_items[self.i].values
            self.content_matrix = alg.fit_transform(self.content_items[self.t])
            # One may pretty print actual words/n-grams instead of positions in arrays using `tfidf_feature_names = alg.get_feature_names()`

            for acc_name in sorted(self.accuracy_methods):
                if acc_name == 'RMSE':
                    continue
                elif acc_name == 'MAE':
                    continue
                elif acc_name == 'RecallAtPosition':
                    self.items['RecallAtPosition' + alg_name] = np.nan
                    loo = LeaveOneOut()
                    for user in self.items[self.u].unique():  # This loop is computationally expensive
                        user_items = self.items[self.items[self.u] == user]
                        user_item_idx = np.isin(content_item_ids, user_items[self.i])
                        user_features = self.content_matrix[user_item_idx]

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
                            cosine_similarities = cosine_similarity(content_train_users_profiles.reshape(1, -1), self.content_matrix[top_test_idx]).flatten()
                            sorted_similarities_indices = (-1 * cosine_similarities).argsort()
                            sorted_item_ids = content_item_ids[top_test_idx][sorted_similarities_indices]
                            pos = np.where(sorted_item_ids == test_user_feature[0])[0][0]

                            self.items.at[(self.items[self.u] == user) & (self.items[self.i] == test_user_feature[0]), 'RecallAtPosition' + alg_name] = pos

        # Overall accuracy
        for alg_name in sorted(self.algorithms_name):
            log_line = '{:<25s} (overall) ::'.format(alg_name)
            for acc_name in sorted(self.accuracy_methods):
                if acc_name == 'RMSE':
                    continue
                elif acc_name == 'MAE':
                    continue
                elif acc_name == 'RecallAtPosition':
                    overall_acc = self.items['RecallAtPosition' + alg_name].mean()
                else:
                    raise ValueError('Expected a valid name for an accuracy method, got "{}".'.format(acc_name))

                log_line += ' | Overall-{0:s}: {overall_acc:>7.2f}'.format(acc_name, overall_acc=overall_acc)

            self._logger.info(log_line)

        return self

    class GensimFastText(object):
        """Gensim's FastText algorithm adapted for Recommender System tasks.

        Word embedding for document similarity comparisons.

        Methods:
            fit_transform(X[, y]): Fit model to training data and return the estimate for the input - with target y being ignored.
            estimate(X): Predict target using fitted model.
        """

        def __init__(self, fasttext_model_file=None, **kwargs):
            """Initialize internal attributes of the class.

            Args:
                fasttext_model_file: Path to a model file in fastText format.
                kwargs: Dictionary of additional arguments which are passed to the underlying algorithm.
            """
            if fasttext_model_file is None:
                self.ft = FastText(**kwargs)
            else:
                self.ft = FastText.load_fasttext_format(fasttext_model_file, **kwargs)

        def _transform(self, input_set):
            input_set = input_set.str.split(r'\.\!\?; ')
            # Strip empty elements from the lists contained in the table
            input_set = input_set.apply(lambda x: list(filter(None, x)))

            content_matrix = np.zeros((input_set.shape[0], self.ft.wv.vector_size))
            for idx in range(content_matrix.shape[0]):
                content_matrix[idx] = self.ft.wv[input_set.iloc[idx]].sum(axis=0)

            content_matrix = normalize(content_matrix, axis=1)

            return content_matrix

        def fit_transform(self, train_set, y=None):
            """Perform the document feature assimilation via utilizing the gathered word embeddings, store the resulting parameters for later estimations and return the final estimate for the input.

            Args:
                train_set: Dataset used for training.
                y: Ignored optional target variable; present only for compatibility reasons.
                _internal_init_sims: Internally used variable to indicate whether the vectors should be normalized.

            Returns:
                self: The instance of the fitted model.
            """
            # Normalize the feature vectors; this prohibits further training
            self.ft.init_sims(replace=True)

            return self._transform(train_set)

        def estimate(self, test_set):
            """Return an estimate of the given input data using the fit parameters of the model.

            Args:
                test_set: Dataset used for training.

            Returns:
                Estimation of the set using the fitted model for transformation.
            """
            return self._transform(test_set)
