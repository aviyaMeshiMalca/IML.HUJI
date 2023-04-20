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

    # Define percentages to sample from the overall training data
    percentages = np.arange(0.1, 1.01, 0.01)

    # Define number of iterations for each percentage
    num_iter = 10

    # Define arrays to store results
    avg_loss = np.zeros_like(percentages)
    std_loss = np.zeros_like(percentages)

    # Define linear regression model
    model = LinearRegression(include_intercept=True)

    # Loop over each percentage
    for i, p in enumerate(percentages):
        # Initialize array to store losses for each iteration
        losses = np.zeros(num_iter)
        
        # Repeat for num_iter times
        for j in range(num_iter):
            # Sample p% of the overall training data
            preprocessed_X, preprocessed_y = preprocess_data(X.copy(), y.copy())
            (preprocessed_train_X, preprocessed_train_y, preprocessed_test_X, preprocessed_test_y) = split_train_test(preprocess_X, preprocess_y, p)
            
            # (train_X, train_y, test_X, test_y) = split_train_test(X, y, p)
            # preprocessed_train_X, preprocessed_train_y = preprocess_data(train_X, train_y)
            # preprocessed_test_X, preprocessed_test_y = preprocess_data(test_X, test_y)
            
            # Fit linear model over sampled set
            model.fit(preprocessed_train_X, preprocessed_train_y)
            
            # Calculate loss over test set
            loss = model._loss(preprocessed_test_X, preprocessed_test_y )
            losses[j] = loss
        
        # Store average and variance of loss over test set for current percentage
        avg_loss[i] = np.mean(losses)
        std_loss[i] = np.std(losses)
        
    # Plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    plt.errorbar(percentages, avg_loss, yerr=2*std_loss, fmt='-o')
    plt.xlabel('Percentage of overall training data')
    plt.ylabel('Average square loss over test set')
    plt.show()


