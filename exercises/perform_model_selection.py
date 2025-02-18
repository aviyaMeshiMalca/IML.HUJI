from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 1 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_X, train_y = X[:n_samples], y[:n_samples]
    # test_X, test_y = X[n_samples:], y[n_samples:]

    # train_X, train_y = train_X.reshape(-1, 1), train_y.reshape(-1, 1)
    # val_X, val_y = val_X.reshape(-1, 1), val_y.reshape(-1, 1)

    # Question 2 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    values = np.linspace(start=0, stop=5, num=n_evaluations)

    ridge_errors = []
    lasso_errors = []
    for val in values:
        ridge = RidgeRegression(lam=val)
        lasso = Lasso(alpha=val)
    ridge_errors.append(np.mean(
        cross_validate(ridge, train_X, train_y, scoring=mean_square_error)))
    lasso_errors.append(np.mean(
        cross_validate(lasso, train_X, train_y, scoring=mean_square_error)))

    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model


if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()
