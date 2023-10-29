import numpy as np


def compute_MSE(y, tx, w):
    loss = (y - np.dot(tx, w))**2 / (2*len(y))
    return np.sum(loss)
    

def compute_gradient(y, tx, w):
    N = len(y)
    error = y - np.dot(tx, w)
    gradient = -np.dot(tx.T, error) / N
    return gradient


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    loss = compute_MSE(y, tx, w)
    for n_iter in range(max_iters):
        # Compute gradient and loss
        gradient = compute_gradient(y, tx, w)
        # Update w by gradient
        w = w - gamma*gradient
        loss = compute_MSE(y, tx, w)


        print(
            "GD iter. {bi}/{ti}: loss={l}, w={w}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w=w
            )
        )

    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):

    w = initial_w
    loss = compute_MSE(y, tx, w)


    for n_iter in range(max_iters):
        # Select one random data point 
        index = np.random.randint(0, len(y), 1)
        y_batch = y[index]
        tx_batch = tx[index]
        gradient = compute_gradient(y_batch, tx_batch, w)
        w = w - gamma*gradient
        # Compute loss
        loss = compute_MSE(y, tx, w)

        print(
            "SGD iter. {bi}/{ti}: loss={l}, w={w}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w=w
            )
        )
    return w, loss



def least_squares(y, tx):
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_MSE(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    lambda_ = lambda_ * 2 * tx.shape[0]
    w = np.linalg.solve(tx.T.dot(tx) + lambda_ * np.identity(tx.shape[1]), tx.T.dot(y))
    loss = compute_MSE(y, tx, w)
    return w, loss


###### LOGISTIC REGRESSION ######
def sigmoid(t):
    return 1.0 / (1 + np.exp(-t))

def compute_logistic_loss(y, tx, w):
    p = sigmoid(np.dot(tx, w))
    return -1/y.shape[0] * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def compute_logistic_gradient(y, tx, w):
    return 1/y.shape[0] * tx.T.dot(sigmoid(tx.dot(w))- y)

    

def logistic_regression(y, tx, initial_w, max_iters, gamma, print_=True):   # we added print_ to avoid printing when we doing long iterations
    w = initial_w
    loss = compute_logistic_loss(y, tx, w)

    for n_iter in range(max_iters):
        gradient = compute_logistic_gradient(y, tx, w)
        w = w - gamma*gradient
        loss = compute_logistic_loss(y, tx, w)

        if print_:
            print(
                "Logistic Regression iter. {bi}/{ti}: loss={l}, w={w}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss, w=w
                )
            )
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, to_print = True):
    w = initial_w
    loss = compute_logistic_loss(y, tx, w) 

    for n_iter in range(max_iters):
        gradient = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma*gradient
        loss = compute_logistic_loss(y, tx, w) 
        if to_print:
            print(
                "Regularized Logistic Regression iter. {bi}/{ti}: loss={l}, w={w}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss - lambda_ * np.linalg.norm(w)**2, w=w
                )
            )
    return w, loss


