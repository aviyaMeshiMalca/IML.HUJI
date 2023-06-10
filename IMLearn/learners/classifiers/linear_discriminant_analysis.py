from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features' covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)  # we want just one appearance of any class
        n_features = X.shape[1]

        self.mu_ = np.zeros((len(self.classes_), n_features))
        self.cov_ = np.zeros((n_features, n_features))

        self.pi_ = np.zeros(len(self.classes_))

        for cls, X_cls_rows in zip(self.classes_, (X[y == cls] for cls in self.classes_)):
            class_index = np.where(self.classes_ == cls)[0][0]
            self.pi_[class_index] = len(X_cls_rows) / len(X)
            self.mu_[class_index] = np.mean(X_cls_rows, axis=0)
            self.cov_ += (np.cov(X_cls_rows, rowvar=False) * (len(X_cls_rows) - 1) / (len(X) - len(self.classes_)))
        self._cov_inv = np.linalg.inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.classes_[np.argmax(np.dot(X, np.matmul(self._cov_inv, self.mu_.T)) - 1 / 2 * np.diag(
            np.dot(self.mu_, np.matmul(self._cov_inv, self.mu_.T))) + np.log(self.pi_), axis=1)]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        likelihoods = np.zeros((X.shape[0], self.classes_.shape[0]))

        for i in range(self.classes_.shape[0]):
            diff = X - self.mu_[i]
            cov_det = np.linalg.det(self.cov_)
            exp = np.sum(diff * np.dot(diff, self._cov_inv), axis=1) * -1 / 2
            likelihoods[:, i] = np.exp(exp) / (np.sqrt((2 * np.pi) ** self.mu_.shape[1] * cov_det))

        return likelihoods

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X))
