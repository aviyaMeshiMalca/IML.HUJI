import sys
from typing import Tuple

import numpy as np
import seaborn as sns

sys.path.append("../")
from IMLearn.utils import utils as ut
from IMLearn.learners.regressors import linear_regression
from utils import *
from scipy.stats import multivariate_normal as mvn
from sklearn.linear_model import LinearRegression
# from IMLearn.learners.regressors.linear_regression import LinearRegression
from sklearn.metrics import mean_squared_error
from exercises.house_price_prediction import preprocess_data
from exercises.house_price_prediction import feature_evaluation
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

  
import pandas as pd
from typing import Tuple
import os
import matplotlib.pyplot as plt
from exercises import city_temperature_prediction


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of city temperature dataset
    df = city_temperature_prediction.load_data("datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_data = df[df['Country'] == 'Israel']
    fig, ax = plt.subplots(figsize=(10, 8))

    for year, group in israel_data.groupby(israel_data['Date'].dt.year):
        ax.scatter(group['DayOfYear'], group['Temp'], label=str(year), alpha=0.7)

    ax.legend(title='Year')
    ax.set_xlabel('Day of Year')
    ax.set_ylabel('Temperature (Celsius)')
    ax.set_title('Average Daily Temperature by Day of Year for Israel')
    plt.show()

    # Group the data by month and compute the standard deviation of the daily temperatures
    israel_data_monthly = \
        israel_data.groupby(israel_data['Date'].dt.month)['Temp'].agg(['std'])
    # Create a bar plot of the standard deviation by month
    fig, ax = plt.subplots(figsize=(10, 8))
    israel_data_monthly.plot(kind='bar', ax=ax, color='blue')
    ax.set_xlabel('Month')
    ax.set_ylabel('Standard Deviation of Daily Temperature (Celsius)')
    ax.set_title('Variability of Daily Temperature by Month for Israel')
    plt.show()
    
    # Question 3 - Exploring differences between countries
    grouped = df.groupby(['Country', 'Month']).agg({'Temp': ['mean', 'std']})
    grouped = grouped.reset_index()
    # Plot a line plot of the average monthly temperature with error bars
    sns.set_style('whitegrid')
    g = sns.relplot(data=grouped, x='Month', y=('Temp', 'mean'), hue='Country', kind='line', err_style='bars', ci='sd', aspect=1.5)
    g.set_axis_labels('Month', 'Average Temperature (°C)')
    plt.title('Average Monthly Temperature by Country')
    plt.show()


    # Question 4 - Fitting model for different values of `k`
    israel_data = df[df['Country'] == 'Israel']

    X = israel_data['DayOfYear']
    y = israel_data['Temp']
    X.reset_index(inplace=True, drop=True)
    y.reset_index(inplace=True, drop=True)

    (train_X, train_y, test_X, test_y) = split_train_test(X.copy(), y.copy(), 0.75)

    losses = []
    for k in range(1, 11):
        # Fit a polynomial model of degree k to the training data
        model = PolynomialFitting(k)
        model._fit(train_X.to_numpy(), train_y.to_numpy())
    
        # Evaluate the model on the test data
        loss = model._loss(test_X.copy(), test_y)
        losses.append(round(loss, 2))
        print(f"Degree {k}, loss: {losses[-1]}")
    # Plot a bar plot of the test errors for each value of k
    plt.bar(range(1, 11), losses)
    plt.xlabel('Polynomial Degree (k)')
    plt.ylabel('loss')
    plt.title('loss vs Polynomial Degree')
    plt.show()


    # Question 5 - Evaluating fitted model on different countries
    other_countries = df[df['Country'] != 'Israel']['Country'].unique()
    errors = []
    for country in other_countries:
        country_df = df[df['Country'] == country]
        X = country_df['DayOfYear']
        y = country_df['Temp']
        X.reset_index(inplace=True, drop=True)
        y.reset_index(inplace=True, drop=True)

        (train_X, train_y, test_X, test_y) = split_train_test(X.copy(), y.copy(), 0.75)
        model = PolynomialFitting(5)
        model._fit(train_X.to_numpy(), train_y.to_numpy())
        error = model._loss(test_X, test_y)
        errors.append(round(error, 2))

    plt.bar(other_countries, errors)
    plt.xlabel('Country')
    plt.ylabel('MSE')
    plt.title('Model Error on Other Countries than Israel')
    plt.show()

