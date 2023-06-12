import numpy as np
#from ...base import BaseEstimator #todo return
from IMLearn import BaseEstimator #todo delete
from typing import Callable, NoReturn


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations

    self.weights_: List[float]
        List of weights for each fitted estimator, fitted along the boosting iterations

    self.D_: List[np.ndarray]
        List of weights for each sample, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_samples = X.shape[0]
        self.models_ = []
        self.weights_ = []
        self.D_ = []
        D = np.ones(n_samples) / n_samples

        for t in range(self.iterations_):
            model = self.wl_()
            model.fit(X, y * D)
            y_pred = model.predict(X)
            epsilon_t = np.sum(D[y_pred != y])
            # if epsilon_t > 0.5:
            #     raise ValueError("epsilon_t > 0.5")
            if epsilon_t == 1:
                raise ValueError("epsilon_t == 1")
            if epsilon_t == 0:
                # print("epsilon_t == 0 at iteration : ", t)
                w_t = 1
                self.weights_.append(w_t)
                self.models_.append(model)
                self.D_.append(D)
                return
            else:
                w_t = 0.5 * np.log((1 / epsilon_t) - 1)
            if w_t < 0:
                raise ValueError("w_t < 0")
            self.weights_.append(w_t)
            self.models_.append(model)
            self.D_.append(D)
            D = D * np.exp(-y * w_t * y_pred)
            D /= np.sum(self.D_)  #

        # print("weights : ", self.weights_)
        # , "D : ", self.D_)

    def _predict(self, X):
        """
        Predict responses for given samples using fitted estimator over all boosting iterations

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.partial_predict(X, self.iterations_)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function over all boosting iterations

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
        return self.partial_loss(X, y, self.iterations_)

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators up to T learners

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if T > self.iterations_:
            raise ValueError("T cannot be bigger than iterations number!")
        else:
            n_samples = X.shape[0]
            predictions = np.zeros(n_samples)
            for i in range(T):
                predictions += self.weights_[i] * (self.models_[i]).predict(X)
            return np.sign(predictions)

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function using fitted estimators up to T learners

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ..metrics import misclassification_error
        return misclassification_error(y, self.partial_predict(X, T))