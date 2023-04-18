import sys
from typing import Tuple

import numpy as np

sys.path.append("../")
from IMLearn.utils import utils as ut
from IMLearn.learners.regressors import linear_regression
from utils import *
from scipy.stats import multivariate_normal as mvn
# from sklearn.linear_model import LinearRegression
from IMLearn.learners.regressors.linear_regression import LinearRegression
from sklearn.metrics import mean_squared_error
from exercises.house_price_prediction import preprocess_data
from exercises.house_price_prediction import feature_evaluation
from IMLearn.utils import split_train_test
  
import pandas as pd
from typing import Tuple
import os
import matplotlib.pyplot as plt




if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    y = df['price']
    X = df.drop('price', axis=1)
    (train_X, train_y, test_X, test_y) = split_train_test(X, y, 0.75)

    # Question 2 - Preprocessing of housing prices dataset
    preprocess_X, preprocess_y = preprocess_data(train_X, train_y)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(preprocess_X, preprocess_y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    for p in range(10, 101):
        avg_loss = 0
        for i in range(0, 10):
            (train_X, train_y, test_X, test_y) = split_train_test(X, y, p/100)
            preprocess_X, preprocess_y  = preprocess_data(train_X, train_y)
            lr = LinearRegression()
            lr.__init__()
            lr.fit(preprocess_X, preprocess_y)
            avg_loss += lr._loss(test_X, test_y)
        avg_loss /= 10

    # Create scatter plot
        fig, ax = plt.subplots()
        ax.scatter(p, avg_loss, alpha=0.5)
        ax.set_xlabel("sample percantage")
        ax.set_ylabel("average loss")
        ax.set_title(f"average loss as function of sample percantage")

        # Save plot to file
        filename = "average loss as function of sample percantage.png"
        filepath = os.path.join(output_path, filename)
        plt.savefig(filepath)
        plt.close(fig)

    
    # for p in range(10, 100):
    #     (train_X, train_y, test_X, test_y) = split_train_test(X, y, p)
    #     linear_regression.fit()

