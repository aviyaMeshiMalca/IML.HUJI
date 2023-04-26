import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    temperature_df = pd.read_csv(filename, parse_dates=['Date'], na_values=['-'])

    # Convert the column to float
    temperature_df['Temp'] = pd.to_numeric(temperature_df['Temp'], errors='coerce')

    # # Drop rows with NaN values in the column
    # temperature_df.dropna(subset=['Temp'], inplace=True)

    temperature_df.dropna(inplace=True)

    # Add a DayOfYear column
    temperature_df['DayOfYear'] = temperature_df['Date'].dt.dayofyear

    #check that there are no null
    # create a boolean mask of null values
    mask = temperature_df.isnull().any(axis=1)
    temperature_df = temperature_df.loc[~mask]

    #check that cols which soppused to have 
    # only positive values actually make it
    pos_val_X_cols = ["Year","Month","Day"]
    valid_positive_rows = (temperature_df[pos_val_X_cols] >= 0).all(axis=1)
    temperature_df = temperature_df.loc[valid_positive_rows]

    min_temp = -30
    max_temp = 60
    temperature_df = temperature_df[temperature_df["Temp"] >= min_temp]
    temperature_df = temperature_df[temperature_df["Temp"] <= max_temp]

    return temperature_df
    

if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    (X, y) = load_data("datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    raise NotImplementedError()

    # Question 3 - Exploring differences between countries
    raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
    raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    raise NotImplementedError()