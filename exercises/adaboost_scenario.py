import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly as plt


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    print("start qu1")
    adaboost = AdaBoost(wl=DecisionStump, iterations=n_learners)
    adaboost.fit(train_X, train_y)
    train_losses = []
    test_losses = []
    for t in range(n_learners):
        train_losses.append(adaboost.partial_loss(X=train_X, y=train_y, T=t))
        test_losses.append(adaboost.partial_loss(X=test_X, y=test_y, T=t))

    print("finish iteration")
    print(train_X, train_y, test_X, test_y)

    fig = go.Figure()
    training_errors = go.Scatter(x=list(range(0, len(train_losses))), y=train_losses, name='training errors')
    test_errors = go.Scatter(x=list(range(0, len(test_losses))), y=test_losses, name='test errors')
    fig.add_trace(training_errors)
    fig.add_trace(test_errors)
    fig.show()
    # fig.write_image("qu1.png")

    # plt.plot(x = list(range(1, n_learners+1)), y = train_losses, label='training errors')
    # plt.plot(list(range(1, n_learners+1)), test_losses, label='test errors as a function of the number of fitted learners')
    # plt.title("training- and test errors as a function of the number of fitted learners")
    # plt.xlabel("number of fitted learners")
    # plt.ylabel("error")

    print("finish qu1")

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    fig = go.Figure()
    for t in T:
        predict_func = lambda X: adaboost.partial_predict(X, t)
        decision_boundary = decision_surface(predict_func, xrange=lims[0], yrange=lims[1], density=120, dotted=False)
        test_data = go.Scatter(x=test_X[:, 0], y=test_X[:, 1], name='Test Set')
        fig.add_trace(decision_boundary)
        fig.add_trace(test_data)

    fig.update_layout(title='Decision Boundaries', xaxis_title='First feature ', yaxis_title='Second Feature')
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    best_ensemble_size = 250
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    fig = go.Figure()
    predict_func = lambda X: adaboost.partial_predict(X, best_ensemble_size)
    decision_boundary = decision_surface(predict_func, xrange=lims[0], yrange=lims[1], density=120, dotted=False)
    test_data = go.Scatter(x=test_X[:, 0], y=test_X[:, 1], name='Test Set')
    fig.add_trace(decision_boundary)
    fig.add_trace(test_data)

    fig.update_layout(title='Decision Boundaries', xaxis_title='First feature ', yaxis_title='Second Feature')
    fig.show()
    print("QU 3 finish")

    # Question 4: Decision surface with weighted samples
    decision_boundary = decision_surface(predict_func, xrange=lims[0], yrange=lims[1], density=120, dotted=False)
    train_data = go.Scatter(x=train_X[:, 0], y=train_X[:, 1], name='Train Set')
    fig.add_trace(decision_boundary)
    fig.add_trace(train_data)

    fig.update_layout(title='Decision Boundaries', xaxis_title='First feature ', yaxis_title='Second Feature')
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
