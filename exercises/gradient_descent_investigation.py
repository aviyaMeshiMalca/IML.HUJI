import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go

from utils import custom


class Callback:
    def __init__(self):
        self.values_list = list()
        self.weights_list = list()

    def callback_function(self, solver: GradientDescent, weights: np.ndarray, val: np.ndarray,
                          grad: np.ndarray, t: int, eta: float, delta: float):
        self.values_list.append(float(val))
        self.weights_list.append(weights)


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

    # print("shape: ", descent_path.shape)

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

    figL1 = go.Figure()
    figL2 = go.Figure()

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

            fig = plot_descent_path(module=module, descent_path=descent_path,
                                     title="of Module :{}, eta : {}".format(module.__name__, eta))
            fig.show(renderer='browser')

            # 3 For each of the modules, plot the convergence rate (i.e. the norm as a function of the GD
            # iteration) for all specified learning rates. Explain your results
            # module_convergence_norm = [np.linalg.norm(weights) for weights in weights_list]

            if module == L2:
                figure = figL2
            elif module == L1:
                figure = figL1
            else:
                raise ValueError("unknown module")

            figure.add_trace(go.Scatter(x=list(range(len(curr_callback.values_list))),
                                        y=curr_callback.values_list, name=str(eta)))

    figL1.update_layout(title="Convergence Rate for Module L1 with Different Learning Rates",
                        xaxis_title="GD Iteration",
                        yaxis_title="Norm of Weights")

    figL2.update_layout(title="Convergence Rate for Module L2 with Different Learning Rates",
                        xaxis_title="GD Iteration",
                        yaxis_title="Norm of Weights")

    figL1.show(renderer='browser')
    figL2.show(renderer='browser')


# not necessary to submit!!
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

    from IMLearn.learners.classifiers import LogisticRegression

    logistic_regression = LogisticRegression(solver=GradientDescent(max_iter=2000))
    logistic_regression.fit(X_train, y_train)

    # 8. Fit logistic regression model and plot ROC curve
    y_pred_prob = logistic_regression.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve (area = {:.2f})'.format(roc_auc)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
    fig.update_layout(
        xaxis=dict(title='False Positive Rate'),
        yaxis=dict(title='True Positive Rate'),
        title='Receiver Operating Characteristic',
        showlegend=True
    )
    fig.show(renderer='browser')

    # 9. Find the optimal alpha and calculate the model's test error
    alpha_values = np.linspace(0, 1, 100)
    tpr_minus_fpr = []
    for alpha_vlaue in alpha_values:
        y_prediction = np.array(list(map(int, y_pred_prob >= alpha_vlaue)))
        true_negative, false_positive, false_negative, true_positive = confusion_matrix(y_test, y_prediction).ravel()
        tpr_minus_fpr.append(true_positive / (true_positive + false_negative) - false_positive /
                             (false_positive + true_negative))
    optimal_alpha = alpha_values[np.argmax(tpr_minus_fpr)]
    y_pred_optimal = np.array(list(map(int, y_pred_prob >= optimal_alpha)))
    test_error = 1 - accuracy_score(y_test, y_pred_optimal)
    print(f"Optimal alpha:{optimal_alpha}")
    print(f"Model's test error: {test_error}")


if __name__ == '__main__':
    np.random.seed(0)
    # compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
