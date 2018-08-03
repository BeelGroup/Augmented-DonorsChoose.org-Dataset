import numpy as np
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors


def rmse(estimate, truth):
    estimate = np.asarray(estimate[truth.nonzero()]).flatten()
    truth = np.asarray(truth[truth.nonzero()]).flatten()
    return np.sqrt(mean_squared_error(estimate, truth))


def mae(estimate, truth):
    estimate = np.asarray(estimate[truth.nonzero()]).flatten()
    truth = np.asarray(truth[truth.nonzero()]).flatten()
    return mean_absolute_error(estimate, truth)


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
