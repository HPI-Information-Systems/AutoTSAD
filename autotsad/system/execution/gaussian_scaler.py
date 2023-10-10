import warnings
from typing import Optional

import numpy as np
from scipy import sparse
from scipy.special import erf, erfinv
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES, check_array


class GaussianScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Scale features to probabilities assuming a Gaussian distribution.

    This estimator scales and translates each feature individually such that it is in
    the range between 0 and 1 inclusively. Out-of-range values are automatically clipped.

    The transformation is given by::

        X_scaled = erf((X - X.mean(axis=0)) / (X.std(X, axis=0) * np.sqrt(2)))

    .. warning::

        Regularization is not thoroughly tested; use on your own risk!

    .. warning::

        This scaler cannot inverse all transformations because it performs clipping,
        which is not invertible!

    Parameters
    ----------
    regularize : str, optional (default=None)
        Regularization method to use. If None, no regularization is performed.
        Possible values are: ``"inverse-log"``, ``"inverse-linear"``, ``"regular"``.

    base_scores : array-like of shape (n_features,), optional (default=None)
        Base scores to use for regularization. If None, the base scores are set to
        the median of the respective feature. You can set individual base scores to
        ``np.nan``. For those features the base score is set using the default method
        (median).

    copy : bool, optional (default=True)
        If False, try to avoid a copy and do inplace scaling instead. This is not
        guaranteed to always work inplace; e.g. if the data is not a NumPy array or
        scipy.sparse CSR matrix, a copy may still be returned.

    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
        The mean value for each feature in the training set.

    std_ : ndarray of shape (n_features,)
        The standard deviation for each feature in the training set.

    max_ : ndarray of shape (n_features,)
        The maximum value for each feature in the training set.

    base_scores_ : ndarray of shape (n_features,)
        The base score for each feature in the training set.

    See Also
    --------
    Hans-Peter Kriegel, Peer Kroger, Erich Schubert, and Arthur Zimek: Interpreting and Unifying Outlier Scores. In:
    Proceedings of the SIAM International Conference on Data Mining (SDM). 2011.
    `<https://doi.org/10.1137/1.9781611972818.2>`_
    """

    _parameter_constraints: dict = {
        "regularize": [str, None],
        "base_scores": ["array-like", None],
        "copy": ["boolean"],
    }

    def __init__(self, regularize: Optional[str] = None,
                 base_scores: Optional[np.ndarray] = None,
                 copy: bool = True):
        self.regularize = regularize
        self.base_scores = base_scores
        self.copy = copy

    def _regularize(self, X: np.ndarray) -> np.ndarray:
        if self.regularize == "inverse-log":
            if np.min(X, axis=0) > 0:
                return - np.log(X / np.max(X, axis=0))
            else:
                warnings.warn(
                    "Logarithmic inverse scaling is not possible for values <= 0. "
                    "Using linear inverse scaling instead."
                )
                self.regularize = "inverse-linear"

        if self.regularize == "inverse-linear":
            return self.max_ - X

        result = X - self.base_scores_
        result[result < 0] = 0
        return result

    def fit(self, X, y=None):
        """Fit the regularization parameters and the Gaussian distribution to the data
        that are later used for scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to fit the per-feature Gaussian distribution used for later
            scaling along the features' axis.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        self._validate_params()
        if self.regularize is not None and \
                self.regularize not in ["inverse-log", "inverse-linear", "regular"]:
            raise ValueError(f"Unknown regularization method '{self.regularize}'!")

        if sparse.issparse(X):
            raise TypeError("MinMaxScaler does not support sparse input.")

        X = self._validate_data(
            X,
            reset=True,
            dtype=FLOAT_DTYPES,
            force_all_finite=True,
        )

        # for regularization
        self.max_ = np.max(X, axis=0)
        if self.base_scores is not None:
            if self.base_scores.shape[0] != X.shape[1]:
                raise ValueError(
                    f"Number of base scores ({self.base_scores.shape[0]}) "
                    f"does not match number of features ({X.shape[1]})!"
                )
            bs = self.base_scores
            if np.any(np.isnan(bs)):
                bs[np.isnan(bs)] = np.median(X[:, np.isnan(bs)], axis=0)
            self.base_scores_ = bs

        else:
            self.base_scores_ = np.median(X, axis=0)

        # for gaussian scaling
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        """Scale features of X according to feature_range.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        check_is_fitted(self)

        X = self._validate_data(
            X,
            copy=self.copy,
            dtype=FLOAT_DTYPES,
            force_all_finite=True,
            reset=False,
        )

        if self.regularize is not None:
            X = self._regularize(X)

        X = erf((X - self.mean_) / (self.std_ * np.sqrt(2)))
        X[X < 0] = 0
        return X

    def inverse_transform(self, X):
        """Undo the scaling of X according to feature_range. This scaler cannot inverse
        all transformations because it performs clipping, which is not invertible.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed. It cannot be sparse.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        check_is_fitted(self)

        X = check_array(
            X, copy=self.copy, dtype=FLOAT_DTYPES, force_all_finite=True
        )

        if self.regularize is not None:
            raise ValueError("Inverse transform is not possible with regularization!")

        X = erfinv(X) * self.std_ * np.sqrt(2) + self.mean_
        return X


if __name__ == '__main__':
    X = np.random.rand(10).reshape(-1, 1)*2+0.5
    Y = np.random.rand(10).reshape(-1, 1)*2+0.9

    # scaler = GaussianScaler()
    # scaler = StandardScaler()
    scaler = MinMaxScaler(clip=True)
    X = scaler.fit_transform(X)
    print(X)
    Y_hat = scaler.transform(Y)
    print(Y_hat)
    Y_inv = scaler.inverse_transform(Y_hat)
    print(np.c_[Y, Y_inv])
    np.testing.assert_array_almost_equal(Y, Y_inv, 4)
