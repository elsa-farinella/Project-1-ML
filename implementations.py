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
        #compute gradient and loss
        gradient = compute_gradient(y, tx, w)
        #update w by gradient
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
        #compute gradient for 1 point and then update the weights
        index = np.random.randint(0, len(y), 1)
        y_batch = y[index]
        tx_batch = tx[index]
        gradient = compute_gradient(y_batch, tx_batch, w)
        w = w - gamma*gradient
        #compute loss
        loss = compute_MSE(y, tx, w)

        print(
            "SGD iter. {bi}/{ti}: loss={l}, w={w}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w=w
            )
        )
    return w, loss



def least_squares(y, tx):
    # returns mse, and optimal weights
    # ***************************************************
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

# ricordiamo che le classi sono -1 e 1 quindi dobbiamo adattare la loss di conseguenza
def compute_logistic_loss(y, tx, w, K):
    #return 1/y.shape[0] * np.sum(np.log(1 + np.exp(-y * tx.dot(w))))
    p = sigmoid(np.dot(tx, w))
    #return 1/y.shape[0] * np.sum(np.log(1 + np.exp(tx.dot(w))) - y * tx.dot(w))
    return -1/y.shape[0] * np.sum(y * np.log(p) + K * (1 - y) * np.log(1 - p))


def compute_logistic_gradient(y, tx, w, K):
    # passo anche k, peso della predizione negativa e calcolo il gradiente di conseguenza
    #return -1/y.shape[0] * (tx.T *sigmoid(-y.T.dot(tx.dot(w)))).dot(y)
    return 1/y.shape[0] * tx.T.dot(sigmoid(tx.dot(w))- y)

    

def logistic_regression(y, tx, initial_w, max_iters, gamma, print_=True, K=1):
    w = initial_w
    loss = compute_logistic_loss(y, tx, w, K)

    for n_iter in range(max_iters):
        gradient = compute_logistic_gradient(y, tx, w, K)
        w = w - gamma*gradient
        loss = compute_logistic_loss(y, tx, w, K)

        if print_:
            print(
                "Logistic Regression iter. {bi}/{ti}: loss={l}, w={w}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss, w=w
                )
            )
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, to_print = True, K=1):
    w = initial_w
    loss = compute_logistic_loss(y, tx, w, K) 

    for n_iter in range(max_iters):
        gradient = compute_logistic_gradient(y, tx, w, K) + 2 * lambda_ * w
        w = w - gamma*gradient
        loss = compute_logistic_loss(y, tx, w, K) 
        if to_print:
            print(
                "Regularized Logistic Regression iter. {bi}/{ti}: loss={l}, w={w}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss - lambda_ * np.linalg.norm(w)**2, w=w
                )
            )
    return w, loss




    ##### TEST #####


# initial_w = np.array([0.5, 1.0])
# y = np.array([0.1, 0.3, 0.5])
# tx = np.array([[2.3, 3.2], [1.0, 0.1], [1.4, 2.3]])
# MAX_ITERS = 2
# GAMMA = 0.1
# RTOL = 1e-4
# ATOL = 1e-8

# def test_mean_squared_error_gd_0_step(y, tx):
#     expected_w = np.array([0.413044, 0.875757])
#     w, loss = mean_squared_error_gd(y, tx, expected_w, 0, GAMMA)

#     expected_w = np.array([0.413044, 0.875757])
#     expected_loss = 2.959836

#     np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
#     np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
#     assert loss.ndim == 0
#     assert w.shape == expected_w.shape

# def test_mean_squared_error_gd(y, tx, initial_w):
#     w, loss = mean_squared_error_gd(
#         y, tx, initial_w, MAX_ITERS, GAMMA
#     )

#     expected_w = np.array([-0.050586, 0.203718])
#     expected_loss = 0.051534

#     np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
#     np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
#     assert loss.ndim == 0
#     assert w.shape == expected_w.shape

# #test_mean_squared_error_gd_0_step(y, tx)

# #test_mean_squared_error_gd(y, tx, initial_w)


# def test_mean_squared_error_sgd(y, tx, initial_w):
#     # n=1 to avoid stochasticity
#     w, loss = mean_squared_error_sgd(
#         y[:1], tx[:1], initial_w, MAX_ITERS, GAMMA
#     )

#     expected_loss = 0.844595
#     expected_w = np.array([0.063058, 0.39208])

#     np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
#     np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
#     assert loss.ndim == 0
#     assert w.shape == expected_w.shape

# #test_mean_squared_error_sgd(y, tx, initial_w)

# def test_least_squares(y, tx):
#     w, loss = least_squares(y, tx)

#     expected_w = np.array([0.218786, -0.053837])
#     expected_loss = 0.026942

#     np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
#     np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
#     assert loss.ndim == 0
#     assert w.shape == expected_w.shape

# test_least_squares(y, tx)


# def test_ridge_regression_lambda0(y, tx):
#     lambda_ = 0.0
#     w, loss = ridge_regression(y, tx, lambda_)

#     expected_loss = 0.026942
#     expected_w = np.array([0.218786, -0.053837])

#     np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
#     np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
#     assert loss.ndim == 0
#     assert w.shape == expected_w.shape

# test_ridge_regression_lambda0(y, tx)

# def test_ridge_regression_lambda1(y, tx):
#     lambda_ = 1.0
#     w, loss = ridge_regression(y, tx, lambda_)

#     expected_loss = 0.03175
#     expected_w = np.array([0.054303, 0.042713])

#     np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
#     np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
#     assert loss.ndim == 0
#     assert w.shape == expected_w.shape

# test_ridge_regression_lambda1(y, tx)

# def test_logistic_regression_0_step(y, tx):
#     expected_w = np.array([0.463156, 0.939874])
#     y = (y > 0.2) * 1.0
#     w, loss = logistic_regression(y, tx, expected_w, 0, GAMMA)

#     expected_loss = 1.533694

#     np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
#     np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
#     assert loss.ndim == 0
#     assert w.shape == expected_w.shape


# test_logistic_regression_0_step(y, tx)


# def test_logistic_regression(y, tx, initial_w):
#     y = (y > 0.2) * 1.0
#     w, loss = logistic_regression(
#         y, tx, initial_w, MAX_ITERS, GAMMA
#     )

#     expected_loss = 1.348358
#     expected_w = np.array([0.378561, 0.801131])

#     np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
#     np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
#     assert loss.ndim == 0
#     assert w.shape == expected_w.shape

# #test_logistic_regression(y, tx, initial_w)

# def test_reg_logistic_regression(y, tx, initial_w):
#     lambda_ = 1.0
#     y = (y > 0.2) * 1.0
#     w, loss = reg_logistic_regression(
#         y, tx, lambda_, initial_w, MAX_ITERS, GAMMA
#     )

#     expected_loss = 0.972165
#     expected_w = np.array([0.216062, 0.467747])

#     np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
#     np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
#     assert loss.ndim == 0
#     assert w.shape == expected_w.shape

# #test_reg_logistic_regression(y, tx, initial_w)

# def test_reg_logistic_regression_0_step(y, tx):
#     lambda_ = 1.0
#     expected_w = np.array([0.409111, 0.843996])
#     y = (y > 0.2) * 1.0
#     w, loss = reg_logistic_regression(
#         y, tx, lambda_, expected_w, 0, GAMMA
#     )

#     expected_loss = 1.407327

#     np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
#     np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
#     assert loss.ndim == 0
#     assert w.shape == expected_w.shape

# test_reg_logistic_regression_0_step(y, tx)