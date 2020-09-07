from sklearn.neural_network.gradient_enhanced import GENN
import numpy as np

from sklearn.neural_network.gradient_enhanced._utils import goodness_of_fit
from sklearn.neural_network.gradient_enhanced.tests.test_problems import rastrigin


def test_forward_prop():
    """
    Use a very simple network to check to check that
    forward propagation recovers expected results that
    can be computed by hand. Concretely, the following
    network is equivalent to Y = 4 * X
    """
    model = GENN(hidden_layer_sizes=(2, 2), activation='identity')
    hidden_activation = [model.activation] * len(model.hidden_layer_sizes)
    output_activation = ['identity']
    model._a = hidden_activation + output_activation
    model._W = [np.ones((2, 1)), np.ones((2, 2)), np.ones((1, 2))]
    model._b = [np.zeros((2, 1)), np.zeros((2, 1)), np.zeros((1, 1))]
    model._n_x = 1
    model._n_y = 1
    X = np.array([1, 2, 3, 4]).reshape((-1, 1))

    # States
    y_pred = model.predict(X).ravel()
    y_true = 4 * X.ravel()
    assert np.allclose(y_pred, y_true)

    # Partials
    dydx_pred = model.gradient(X).ravel()
    dydx_true = np.array([4, 4, 4, 4])
    assert np.allclose(dydx_pred, dydx_true)


def test_parameter_shape():
    """
    Make sure that parameter initialization
    produces the correct parameter shapes
    """
    X = np.array([1, 2, 3, 4]).reshape((1, -1))
    y = np.array([1, 2, 3, 4]).reshape((1, -1))
    model = GENN(hidden_layer_sizes=(2, 2))
    model._n_x = X.shape[0]
    model._n_y = y.shape[0]
    model._initialize()

    assert model._W[0].shape == (2, 1)
    assert model._W[1].shape == (2, 2)
    assert model._W[2].shape == (1, 2)
    assert model._b[0].shape == (2, 1)
    assert model._b[1].shape == (2, 1)
    assert model._b[2].shape == (1, 1)


# Test on Rastrigin
# Do speed studies: why is it slower than MLPRegressor (if it is)?
# Updated my GENN repo (and notebooks) -> give new version
def test_model_parabola(verbose=False, plot=False):
    """
    Very simple test: fit a parabola. This test ensures that
    the model mechanics are working.
    """
    # Training data
    X = np.array([1, 2, 3, 4, 5]).reshape((-1, 1))
    Y = X ** 2
    J = (2*X).reshape((-1, 1, 1))

    # Basic neural net (no gradient-enhancement)
    model = GENN(hidden_layer_sizes=(3, 3), activation='tanh',
                 num_epochs=1, max_iter=1000, normalize=True,
                 learning_rate_init=0.05, alpha=0.01, gamma=0, verbose=verbose)
    model.fit(X, Y, J)
    if plot:
        goodness_of_fit(model, X, Y, title='NN')
    assert model.score(X, Y) > 0.99

    # Gradient-enhanced neural net
    model = GENN(hidden_layer_sizes=(3, 3), activation='tanh',
                 num_epochs=1, max_iter=1000, normalize=True,
                 learning_rate_init=0.05, alpha=0.01, gamma=1, verbose=verbose)
    model.fit(X, Y, J)
    if plot:
        goodness_of_fit(model, X, Y, title='GENN')
    assert model.score(X, Y) > 0.99


def test_model_rastrigin(verbose=False, plot=False):
    # Domain
    lb = -1.
    ub = 1.5

    # Create full-factorial
    X = []
    n = 5
    x = np.linspace(lb, ub, n)
    for i in range(n):
        for j in range(n):
            X.append([x[i], x[j]])

    X = np.array(X)
    Y, J = rastrigin(X)

    model = GENN(hidden_layer_sizes=(12,), activation='tanh',
                 num_epochs=1, max_iter=2000, normalize=True,
                 is_finite_difference=False,
                 learning_rate='constant', random_state=0, tol=1e-12,
                 learning_rate_init=0.05, alpha=0, gamma=1, verbose=verbose)
    model.fit(X, Y, J)
    if plot:
        goodness_of_fit(model, X, Y)
    assert model.score(X, Y) > 0.99


def test_sinusoid(verbose=False, plot=False):
    f = lambda x: x * np.sin(x)
    df_dx = lambda x: np.sin(x) + x * np.cos(x)
    lb = -np.pi
    ub = np.pi
    m = 4  # number of training examples
    n_x = 1  # number of inputs
    n_y = 1  # number of outputs
    X_train = np.linspace(lb, ub, m).reshape((m, n_x))
    Y_train = f(X_train).reshape((m, n_y))
    J_train = df_dx(X_train).reshape((m, n_x, n_y))
    m = 30  # number of test examples
    X_test = lb + np.random.rand(m, 1).reshape((m, n_x)) * (ub - lb)
    Y_test = f(X_test).reshape((m, n_y))
    J_test = df_dx(X_test).reshape((m, n_x, n_y))
    model = GENN(hidden_layer_sizes=(12,), activation='tanh',
                 num_epochs=1, max_iter=1000, normalize=False,
                 is_finite_difference=False,
                 learning_rate='backtracking', random_state=None, tol=1e-6,
                 learning_rate_init=0.05, alpha=0.1, gamma=0, verbose=verbose)
    model.fit(X_train, Y_train, J_train)
    if plot:
        import matplotlib.pyplot as plt
        goodness_of_fit(model, X_test, Y_test)
        X = np.linspace(lb, ub, 100).reshape((100, n_x))
        Y_true = f(X)
        Y_pred = model.predict(X)
        fig, ax = plt.subplots()
        ax.plot(X, Y_pred, 'b-')
        ax.plot(X, Y_true, 'k--')
        ax.plot(X_test, Y_test, 'ro')
        ax.plot(X_train, Y_train, 'k+', mew=3, ms=10)
        ax.set(xlabel='x', ylabel='y', title='GENN')
        ax.legend(['Predicted', 'True', 'Test', 'Train'])
        plt.show()


if __name__ == "__main__":
    # test_parameter_shape()
    # test_forward_prop()
    # test_model_parabola(verbose=False, plot=True)
    # test_model_rastrigin(verbose=True, plot=False)
    test_sinusoid(verbose=True, plot=True)

    import cProfile
    # cProfile.run('test_model(verbose=False)')
