from implementations import *
from load_data import *
import numpy as np
import datetime
from additional_methods import *

def compute_accuracy(y, tx, w):
    pred = np.dot(tx, w)
    pred[pred <= 0.5] = 0
    pred[pred > 0.5] = 1
    return np.sum(pred == y) / len(y)

# Load data
data, y = load_data()
tr_data, te_data, tr_y, te_y = split_data(data, y, 0.8, seed=1)
tx = build_model_data(tr_data)
test_tx = build_model_data(te_data)


# Define the hyper-parameters
max_iters = 300
gamma = 0.3


# # # Initialize weights for GD
# w_initial = np.ones(tx.shape[1])

# # Start GD.
# start_time = datetime.datetime.now()
# w, loss = mean_squared_error_gd(tr_y, tx, w_initial, max_iters, gamma)
# end_time = datetime.datetime.now()

# # Print result
# exection_time = (end_time - start_time).total_seconds()
# print("SGD: execution time={t:.3f} seconds".format(t=exection_time))
# accuracy = compute_accuracy(tr_y, tx, w)
# print("Accuracy: ", accuracy)


# # test accuracy 
# test_pred = np.dot(test_tx, w)
# test_pred[test_pred <= 0.5] = 0
# test_pred[test_pred > 0.5] = 1

# print("Test accuracy: ", np.sum(test_pred == te_y) / len(te_y))

# #start SGD 
# start_time = datetime.datetime.now()
# w, loss = stochastic_gradient_descent(
#     y, tx, w_initial, max_iters, gamma
# )
# end_time = datetime.datetime.now()

# excution_time = (end_time - start_time).total_seconds()
# print("SGD: execution time={t:.3f} seconds".format(t=excution_time))
# accuracy = compute_accuracy(y, tx, w)
# print("Accuracy: ", accuracy)



# # start least square method
# start_time = datetime.datetime.now()
# w, loss = least_squares(y, tx)
# print(w, loss)
# end_time = datetime.datetime.now()
# execution_time = (end_time - start_time).total_seconds()
# print("LS: execution time={t:.3f} seconds".format(t=execution_time))
# accuracy = compute_accuracy(y, tx, w)
# print("Accuracy: ", accuracy)

# #start ridge regression
# start_time = datetime.datetime.now()
# lambda_ = 0.8
# w, loss = ridge_regression(tr_y, tx, lambda_)
# print(w, loss)
# end_time = datetime.datetime.now()
# execution_time = (end_time - start_time).total_seconds()
# print("RR: execution time={t:.3f} seconds".format(t=execution_time))
# accuracy = compute_accuracy(tr_y, tx, w)
# print("Accuracy: ", accuracy)
# # test accuracy 
# test_pred = np.dot(test_tx, w)
# test_pred[test_pred <= 0.5] = 0
# test_pred[test_pred > 0.5] = 1

# print("Test accuracy: ", np.sum(test_pred == te_y) / len(te_y))


#start logistic regression

# max_iters = 600
# gamma = 0.3
# start_time = datetime.datetime.now()
# w_initial = np.zeros(tx.shape[1])
# w, loss = logistic_regression(tr_y, tx, w_initial, max_iters, gamma)
# pred_y = sigmoid(np.dot(test_tx, w))
# pred_y[pred_y <= 0.5] = 0
# pred_y[pred_y > 0.5] = 1
# print("Test accuracy: ", np.sum(pred_y == te_y) / len(te_y))


# # start reg logistic regression
# max_iters = 600
# gamma = 0.15
# lambda_ = 0.08
# start_time = datetime.datetime.now()
# w_initial = np.zeros(tx.shape[1])
# w, loss = reg_logistic_regression(tr_y, tx, lambda_, w_initial, max_iters, gamma)
# pred_y = sigmoid(np.dot(test_tx, w))
# pred_y[pred_y <= 0.5] = 0
# pred_y[pred_y > 0.5] = 1
# print("Test accuracy: ", np.sum(pred_y == te_y) / len(te_y))


# polynomial regression
# max_iters = 200
# gamma = 0.02
# degree = 7
# start_time = datetime.datetime.now()
# w, loss = polynomial_regression(tr_y, tr_data, degree, max_iters, gamma)
# print(loss)
# end_time = datetime.datetime.now()
# execution_time = (end_time - start_time).total_seconds()
# poly_test_tx = build_poly(te_data, degree)
# pred_y = np.dot(poly_test_tx, w)
# pred_y[pred_y <= 0.5] = 0
# pred_y[pred_y > 0.5] = 1
# print("Test accuracy: ", np.sum(pred_y == te_y) / len(te_y))

# cross validation

# k = 4
# best_w, best_lambda, best_rmse = cross_validation_demo(tr_y, tr_data, 7, k, np.logspace(-4, 0, 30))
# print(best_lambda, best_rmse)
# poly_test_tx = build_poly(te_data, 7)
# pred_y = np.dot(poly_test_tx, best_w)
# pred_y[pred_y <= 0.5] = 0
# pred_y[pred_y > 0.5] = 1
# print("Test accuracy: ", np.sum(pred_y == te_y) / len(te_y))


def cross_validation(y, x, k_indices, k, lambda_, degree):
    k_test_indices = k_indices[k]
    k_train_indices = np.delete(k_indices, k, axis=0).flatten()
    x_train = x[k_train_indices]
    y_train = y[k_train_indices]
    x_test = x[k_test_indices]
    y_test = y[k_test_indices]

    tx_train = build_poly(x_train, degree)
    tx_test = build_poly(x_test, degree)

    w1, loss_tr = ridge_regression(y_train, tx_train, lambda_)
    w, loss_te = ridge_regression(y_test, tx_test, lambda_)

    return w, loss_tr, loss_te

