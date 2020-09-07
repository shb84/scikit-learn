"""
G R A D I E N T - E N H A N C E D   N E U R A L   N E T W O R K S  (G E N N)

Author: Steven H. Berguin <stevenberguin@gmail.com>

This package is distributed under the MIT license.
"""
import numpy as np
import math
from typing import List

from importlib.util import find_spec

if find_spec("matplotlib"):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_INSTALLED = True
else:
    MATPLOTLIB_INSTALLED = False

EPS = np.finfo(float).eps  # small number to avoid division by zero


def mini_batches(X: np.ndarray, batch_size: int,
                 shuffle: bool = True, seed: int = None) -> list:
    """
    Create randomized mini-batches by returning a list of tuples, where
    each tuple contains the indices of the training data points associated with
    that mini-batch

    Parameters
    ----------
        X: np.ndarray
            input features of the training data
            shape = (n_x, m) where m = num of examples and n_x = num of inputs

        batch_size: int
            mini batch size (if None, then batch_size = m)
            
        shuffle: bool 
            Shuffle data points
            Default = True 
            
        seed: int 
            Random seed (set to make runs repeatable)
            Default = None  

    Returns
    -------
        mini_batches: list
            List of mini-batch indices to use for slicing data, where the index
            is in the interval [1, m]
    """

    np.random.seed(seed)

    batches = []
    m = X.shape[1]
    if not batch_size:
        batch_size = m
    batch_size = min(batch_size, m)

    # Step 1: Shuffle the indices
    if shuffle:
        indices = list(np.random.permutation(m))
    else:
        indices = np.arange(m)

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m / batch_size))
    k = 0
    for _ in range(num_complete_minibatches):
        batch = indices[k * batch_size:(k + 1) * batch_size]
        batches.append(tuple(batch))
        k += 1

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % batch_size != 0:
        mini_batch = indices[(k + 1) * batch_size:]
        batches.append(tuple(mini_batch))

    return batches


def finite_diff(params: List[np.ndarray], f: callable, dx: float = 1e-6):
    """
    Compute gradient using central difference

    Parameters
    ----------
    params: List[np.ndarray]
        Point at which to evaluate gradient

    f: callable
        Function handle to use for finite difference

    dx: float
        Finite difference step

    Returns
    -------
    dy_dx: List[np.ndarray]
        Derivative of f w.r.t. x
    """
    grads = list()
    for k, x in enumerate(params):
        n, p = x.shape
        dy = np.zeros((n, p))
        for i in range(0, n):
            for j in range(0, p):
                # Forward step
                x[i, j] += dx
                y_fwd = f(params)
                x[i, j] -= dx

                # Backward step
                x[i, j] -= dx
                y_bwd = f(params)
                x[i, j] += dx

                # Central difference
                dy[i, j] = np.divide(y_fwd - y_bwd, 2 * dx)

        grads.append(dy)

    return grads


def grad_check(x: List[np.ndarray], f: callable, dfdx: callable,
               dx: float = 1e-6, tol: float = 1e-6,
               verbose: bool = True) -> bool:
    """
    Compare analytical gradient against finite difference

    Parameters
    ----------
    x: List[np.ndarray]
        Point at which to evaluate gradient

    f: callable
        Function handle to use for finite difference

    dx: float
        Finite difference step

    tol: float
        Tolerance below which agreement is considered acceptable
        Default = 1e-6

    verbose: bool
        Print output to standard out
        Default = True

    Returns
    -------
    success: bool 
        Returns True iff finite difference and analytical grads agree 
    """
    success = True
    dydx = dfdx(x)
    dydx_FD = finite_diff(x, f, dx)
    for i in range(len(x)):
        numerator = np.linalg.norm(dydx[i].squeeze() - dydx_FD[i].squeeze())
        denominator = np.linalg.norm(dydx[i].squeeze()) + np.linalg.norm(
            dydx_FD[i].squeeze())
        difference = numerator / (denominator + EPS)
        if difference > tol or numerator > tol:
            success = False
        if verbose:
            if not success:
                print(f"The gradient w.r.t. x[{i}] is wrong")
            else:
                print(f"The gradient w.r.t. x[{i}] is correct")
            print(f"Finite dif: grad[{i}] = {dydx_FD[i].squeeze()}")
            print(f"Analytical: grad[{i}] = {dydx[i].squeeze()}")
            print()
    return success


def goodness_of_fit(model,
                    X: np.ndarray,
                    Y: np.ndarray,
                    J: np.ndarray = None,
                    response: int = 0, partial: int = 0, title: str = None):
    """
    Plot actual by predicted and histogram of prediction error

    Parameters
    ----------
    model: GENN
        The model to be used for prediction

    X: np.ndarray
        Test data inputs
        shape = (m, n_x) where m = no. examples, n_x = no. features

    Y: np.ndarray
        Test data outputs
        shape = (m, n_y) where m = no. examples, n_y = no. outputs

    J: np.ndarray
        Test data outputs
        shape = (m, n_x, n_y)

    response: int
        Index of response to be plotted
        Default = 0

    partial: int
        Index of partial to be plotted. If provided, partial will be plotted.
        If not, response will be plotted.
        Default: None

    title: str
        Title to give the plot
        Default: None
    """
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(title, fontsize=16)
    spec = gridspec.GridSpec(ncols=2, nrows=1, wspace=0.25)

    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[0, 1])

    Y_pred = model.predict(X)
    J_pred = J
    if J:
        J_pred = model.gradient(X)

    def rsquare(y_pred, y_true):
        epsilon = 1e-12  # small number to avoid division by zero
        y_bar = np.mean(y_true)
        SSE = np.sum(np.square(y_pred - y_true))
        SSTO = np.sum(np.square(y_true - y_bar) + epsilon)
        R2 = 1 - SSE / SSTO
        return R2

    if type(J) == np.ndarray:
        dy = response
        dx = partial
        actual = J[:, dy, dx]
        predicted = J_pred[:, dy, dx]
        if not title:
            title = 'Goodness of fit for dY' + str(dy) + '/dX' + str(dx)
    else:
        y = response
        actual = Y[:, y]
        predicted = Y_pred[:, y]
        if not title:
            title = 'Goodness of fit for Y' + str(y)

    metrics = dict()
    metrics['R_squared'] = np.round(rsquare(predicted, actual), 2)
    metrics['std_error'] = np.round(np.std(predicted - actual), 2)
    metrics['avg_error'] = np.round(np.mean(predicted - actual), 2)

    # Reference line
    y = np.linspace(np.min(actual), np.max(actual), 100)

    # Prepare to plot
    if not MATPLOTLIB_INSTALLED:
        raise ImportError("Matplotlib must be installed.")

    ax1.plot(y, y)
    ax1.scatter(actual, predicted, s=100, c='k', marker="+")
    ax1.legend(["perfect fit line", "sample points"])
    ax1.set_xlabel("actual")
    ax1.set_ylabel("predicted")
    ax1.set_title("RSquare = " + str(metrics['R_squared']))

    error = (predicted - actual)
    weights = np.ones(error.shape) / predicted.size
    ax2.hist(error, weights=weights, facecolor='g', alpha=0.75)
    ax2.set_xlabel('Absolute Prediction Error')
    ax2.set_ylabel('Probability')
    ax2.set_title(
        '$\mu$=' + str(metrics['avg_error']) + ', $\sigma=$' + str(
            metrics['std_error']))
    plt.grid(True)
    plt.show()
