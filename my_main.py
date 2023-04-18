import sys
from typing import Tuple

import numpy as np

sys.path.append("../")
from IMLearn.utils import utils as ut
from IMLearn.learners.regressors import linear_regression
from utils import *
from scipy.stats import multivariate_normal as mvn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from exercises.house_price_prediction import preprocess_data
from exercises.house_price_prediction import feature_evaluation
from IMLearn.utils import split_train_test
import pandas as pd
from typing import Tuple
import os

if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    y = df.iloc[:, 2]
    X = df.drop(df.columns[2], axis=1)
    (train_X, train_y, test_X, test_y) = split_train_test(X, y, 0.75)

    # Question 2 - Preprocessing of housing prices dataset
    X, y = preprocess_data(X, y)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    
    # for p in range(10, 100):
    #     (train_X, train_y, test_X, test_y) = split_train_test(X, y, p)
    #     linear_regression.fit()

