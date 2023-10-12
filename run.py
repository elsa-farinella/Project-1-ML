from implementations import *
from load_data import *
import numpy as np
import datetime

def compute_accuracy(y, tx, w):
    pred = np.dot(tx, w)
    pred[pred <= 0.5] = 0
    pred[pred > 0.5] = 1
    return np.sum(pred == y) / len(y)

# Load data
data, y = load_data()
tr_data, te_data, tr_y, te_y = split_data(data, y, 0.8, seed=1)
tx = build_model_data(tr_y, tr_data)
test_tx = build_model_data(te_y, te_data)


# Define the hyper-parameters
max_iters = 200
gamma = 0.1


# # Initialize weights for GD
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
