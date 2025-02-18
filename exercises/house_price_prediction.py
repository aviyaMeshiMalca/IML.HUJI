from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
import os


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    #drop irelevant features
    X.drop(['id','lat', "date", "zipcode"],axis=1, inplace=True)

    # valid_date_rows = (X['date'] != '0')
    # X = X.loc[valid_date_rows]
    # if y is not None:
    #     y = y[valid_date_rows]
    #     y.reset_index(inplace=True, drop=True)

    # X.reset_index(inplace=True, drop=True)

    # X['date'] = pd.to_datetime(X['date'], format='%Y%m%dT%H%M%S')

    # make zipcode numerical notation
    # zipcode_hot_ones = pd.get_dummies(X['zipcode'], prefix='zipcode')
    # X = pd.concat([X.drop('zipcode', axis=1), zipcode_hot_ones], axis=1)

    X = X.apply(pd.to_numeric, errors='coerce')
    
    #drop negative prices in y
    if y is not None:
        y = pd.to_numeric(y, errors='coerce')
        positive_indices = y[y >= 0].index
        y = y[y >= 0]
        X = X.loc[positive_indices]

    #check that there are no null
    # create a boolean mask of null values
    mask = X.isnull().any(axis=1)
    X = X.loc[~mask]
    if y is not None:
        y = y[~mask]

    
    #check that cols which soppused to have only positive values actually make it
    pos_val_X_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'sqft_above',
                           "sqft_basement", "yr_built", "yr_renovated"]
    valid_positive_rows = (X[pos_val_X_cols] >= 0).all(axis=1)
    # Select only the valid rows and columns
    X = X.loc[valid_positive_rows]
    if y is not None:
        y = y[valid_positive_rows]

    #check that rooms size is not too small
    min_sq_size_of_room = 20
    room_size_X_cols = ['sqft_living','sqft_living15','sqft_lot15']
    valid_room_size_rows = (X[room_size_X_cols] >= min_sq_size_of_room).all(axis=1)
    X = X.loc[valid_room_size_rows]
    if y is not None:
        y = y[valid_room_size_rows]

    X.reset_index(inplace=True, drop=True)

    if y is None:
        return X
    
    y.reset_index(inplace=True, drop=True)
    
    return [X, y]


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #calculate Pearson Correlation
    for feature_name in X.columns:
        feature = X[feature_name]
        # Calculate Pearson correlation between feature and response
        x_std = np.std(feature)
        y_std = np.std(y)
        if (x_std == 0 or y_std == 0):
            pearson = 0
        else:
            pearson = np.cov(feature, y)[0,1] / (x_std * y_std)

        # Create scatter plot
        fig, ax = plt.subplots()
        ax.scatter(feature, y, alpha=0.5)
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Response")
        ax.set_title(f"{feature_name} (Pearson Correlation:{pearson})")

        # Save plot to file
        filename = f"{feature_name}.png"
        filepath = os.path.join(output_path, filename)
        plt.savefig(filepath)
        plt.close(fig)

        return None

if __name__ == '__main__':

        np.random.seed(0)
    df = pd.read_csv("datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    y = df['price']
    X = df.drop('price', axis=1)
    (train_X, train_y, test_X, test_y) = split_train_test(X.copy(), y.copy(), 0.75)

    # Question 2 - Preprocessing of housing prices dataset
    preprocess_X, preprocess_y = preprocess_data(train_X, train_y)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(preprocess_X.copy(), preprocess_y.copy())

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
    (train_X, train_y, test_X, test_y) = split_train_test(X.copy(), y.copy())
    
    preprocessed_train_X, preprocessed_train_y = preprocess_data(train_X.copy(), train_y.copy())
    preprocessed_test_X, preprocessed_test_y = preprocess_data(test_X.copy(), test_y.copy())

    # Loop over each percentage
    for i, p in enumerate(percentages):
        # Initialize array to store losses for each iteration
        losses = np.zeros(num_iter)
        # Repeat for num_iter times
        for j in range(num_iter):
            # Sample p% of the overall training data
            sampled_train_X = preprocessed_train_X.sample(frac=p)
            sampled_train_y = preprocessed_train_y[sampled_train_X.index]
            
            # Fit linear model over sampled set
            model = LinearRegression(include_intercept=True)
            model._fit(sampled_train_X, sampled_train_y)
            
            # Calculate loss over test set
            loss = model._loss(preprocessed_test_X, preprocessed_test_y)
            losses[j] = loss
        
        # Store average and variance of loss over test set for current percentage
        avg_loss[i] = np.mean(losses)
        std_loss[i] = np.std(losses)
        
    # Plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    plt.errorbar(percentages, avg_loss, yerr=2*std_loss, fmt='-o')
    plt.xlabel('Percentage of overall training data')
    plt.ylabel('Average square loss over test set')
    plt.show()



   