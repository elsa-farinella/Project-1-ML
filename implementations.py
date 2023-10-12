import numpy as np


def compute_MSE(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    loss = (y - np.dot(tx, w))**2 / (2*len(y))
    return np.sum(loss)
    

def compute_gradient(y, tx, w):
    N = len(y)
    error = y - np.dot(tx, w)
    gradient = -np.dot(tx.T, error) / N
    return gradient


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    # Define parameters to store w and loss

    w = initial_w
    for n_iter in range(max_iters):
        #compute gradient and loss
        loss = compute_MSE(y, tx, w)
        gradient = compute_gradient(y, tx, w)
        #update w by gradient
        w = w - gamma*gradient

        print(
            "GD iter. {bi}/{ti}: loss={l}, w={w}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w=w
            )
        )

    return w, loss


def stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: the final values of the model parameters
        loss: the value of the loss at the last iteration of SGD
    """

    w = initial_w

    for n_iter in range(max_iters):
        #compute loss
        loss = compute_MSE(y, tx, w)
        #compute gradient for 1 point and then update the weights
        index = np.random.randint(0, len(y), 1)
        y_batch = y[index]
        tx_batch = tx[index]
        gradient = compute_gradient(y_batch, tx_batch, w)
        w = w - gamma*gradient

        print(
            "SGD iter. {bi}/{ti}: loss={l}, w={w}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w=w
            )
        )
    return w, loss



def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    # returns mse, and optimal weights
    # ***************************************************
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_MSE(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    # ***************************************************
    lambda_ = lambda_ * 2 * tx.shape[0]
    w = np.linalg.solve(tx.T.dot(tx) + lambda_ * np.identity(tx.shape[1]), tx.T.dot(y))
    loss = compute_MSE(y, tx, w)
    return w, loss
