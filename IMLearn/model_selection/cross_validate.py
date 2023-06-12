from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_scores = []
    validation_scores = []
    indexes = np.arange(len(X))
    fold_size = int(cv / len(X))
    for t in range(cv):
        train_indexes = np.concatenate((indexes[:t * fold_size], indexes[(t + 1) * fold_size:]))
        train_X = X[train_indexes]
        train_y = y[train_indexes]
        validation_indexes = indexes[t * fold_size: (t + 1) * fold_size]
        val_X = X[validation_indexes]
        val_y = y[validation_indexes]

        model = deepcopy(estimator)
        model.fit(train_X, train_y)
        train_scores.append(scoring(train_y, model.predict(train_X)))
        validation_scores.append(scoring(val_y, model.predict(val_X)))

    train_score = np.mean(train_scores)
    validation_score = np.mean(validation_scores)

    return train_score, validation_score
