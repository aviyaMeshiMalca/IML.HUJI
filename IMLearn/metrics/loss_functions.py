import numpy as np


def mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate MSE loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    MSE of given predictions
    """
    if y_true.shape != y_pred.shape:
        raise ValueError('Shapes of y_pred and y_true are different')

    if y_true.size == 0:
        return 0

    arr = np.square(y_true - y_pred)
    return np.mean(arr)


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not

    Returns
    -------
    Misclassification of given predictions
    """
    if y_true.shape != y_pred.shape:
        raise ValueError('Shapes of y_pred and y_true are different')

    if y_true.size == 0:
        return 0

    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)

    num_of_miss_class = 0.0
    for i in range(y_true.size):
        if y_true[i] != y_pred[i]:
            num_of_miss_class += 1

    if normalize:
        return num_of_miss_class / y_true.size

    return num_of_miss_class


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Accuracy of given predictions
    """
    if y_true.shape != y_pred.shape:
        raise ValueError('Shapes of y_pred and y_true are different')

    if y_true.size == 0:
        return 0

    return np.sum(y_true == y_pred) / len(y_true)


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the cross entropy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Cross entropy of given predictions
    """
    raise NotImplementedError()


def softmax(X: np.ndarray) -> np.ndarray:
    """
    Compute the Softmax function for each sample in given data

    Parameters:
    -----------
    X: ndarray of shape (n_samples, n_features)

    Returns:
    --------
    output: ndarray of shape (n_samples, n_features)
        Softmax(x) for every sample x in given data X
    """
    raise NotImplementedError()
