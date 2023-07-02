import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from matplotlib import pyplot as plt

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go


class Callback:
    def __init__(self):
        self.values_list = list()
        self.weights_list = list()

    def callback_function(self, solver: GradientDescent, weights: np.ndarray, val: np.ndarray,
                          grad: np.ndarray, t: int, eta: float, delta: float):
        self.values_list.append(float(val))
        self.weights_list.append(weights)
        # print("callback append w_t to list", weights)
        # print("callback append val to list", val)


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface

    print( "shape: ", descent_path.shape)

    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    callback = Callback()
    return callback.callback_function, callback.values_list, callback.weights_list


def my_get_gd_state_recorder_callback() -> Callback:
    return Callback()


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    # 1 Load and split SA Heart Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    modules = [L1, L2]

    for eta in etas:
        for module in modules:
            curr_callback = my_get_gd_state_recorder_callback()
            lr = FixedLR(eta)
            GD = GradientDescent(learning_rate=lr, callback=curr_callback.callback_function).fit(f=module(init),
                                                                                                 X=X_train.copy(),
                                                                                                 y=y_train.copy())

            if len(curr_callback.weights_list) != len(curr_callback.values_list):
                raise ValueError("len(curr_callback.weights_list) != len(curr_callback.values_list)")

            descent_path = np.array(curr_callback.weights_list)
            print("before pass descent_path :shape: ", descent_path.shape)

            fig = plot_descent_path(module=module, descent_path=descent_path,
                                     title="of Module :{}, eta : {}".format(module.__name__, eta))
            fig.show(renderer='browser')

            # 2 Describe two phenomena that can be seen in the descent path of the ℓ1 objective when using
            # GD and a fixed learning rate.

            # 3 For each of the modules, plot the convergence rate (i.e. the norm as a function of the GD
            # iteration) for all specified learning rates. Explain your results
            # module_convergence_norm = [np.linalg.norm(weights) for weights in weights_list]
            plt.plot(curr_callback.values_list, label="L1")
            plt.xlabel("GD Iteration")
            plt.ylabel("Norm of Weights")
            plt.title("Convergence Rate for Module : {}, with Learning Rate = {}".format(module.__name__, eta))
            plt.legend()
            plt.show()




# not neccesary to submit!!
def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    raise NotImplementedError()

    # Plot algorithm's convergence for the different values of gamma
    raise NotImplementedError()

    # Plot descent path for gamma=0.95
    raise NotImplementedError()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heart Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    raise NotImplementedError()

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    # fit_logistic_regression()
