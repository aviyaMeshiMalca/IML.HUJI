from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet
from numpy import random
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# from scipy import stats
from scipy.stats import norm
from scipy.optimize import curve_fit
# from scipy import asarray as ar,exp
import sys
 

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    X = np.random.normal(10, 1, 1000)
    ge = UnivariateGaussian()
    ge.__init__(False)
    ge.fit(X)

    print((ge.mu_, ge.var_))

    ge._plot_graph(X)

    plt.scatter(X, ge.pdf(X))
    plt.title("PDF - GAUSSIAN") 
    plt.xlabel("sample") 
    plt.ylabel("pdf") 
    plt.show()

    # Question 2 - Empirically showing sample mean is consistent
    

    # Question 3 - Plotting Empirical PDF of fitted model
    

def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = [0,0,4,0]
    sigma = [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0,0,1,0], [0.5, 0, 0, 1]]
    X = np.random.multivariate_normal(mu,sigma,1000)
    ge = MultivariateGaussian()
    ge.__init__()

    ge.fit(X)

    print("mu:\n")
    print(ge.mu_) 
    print("\n")
    print("cov:\n")
    print(ge.cov_)
    print("\n")


    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    log_likelihood = np.array([[ge.log_likelihood([i, 0, j, 0], ge.cov_, X) for i in f1] for j in f3])
    fig, ax = plt.subplots()
    im = ax.imshow(log_likelihood)
    plt.show()

    # Question 5 - Likelihood evaluation
    

    # Question 6 - Maximum likelihood
    
if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
