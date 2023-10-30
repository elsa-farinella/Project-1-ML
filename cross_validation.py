import numpy as np
from implementations import *
from additional_methods import *


def generate_k_fold_indices(y, k_folds, seed):
    
    # Calculate the total number of data points.
    total_data_points = y.shape[0]
    
    # Calculate the number of data points in each fold.
    data_points_per_fold = int(total_data_points / k_folds)
    
    # Set the random seed for reproducibility.
    np.random.seed(seed)
    
    # Generate a random permutation of all indices.
    shuffled_indices = np.random.permutation(total_data_points)
    
    # Split the shuffled indices into 'k_folds' consecutive subsets.
    fold_indices = [shuffled_indices[k * data_points_per_fold: (k + 1) * data_points_per_fold]
                    for k in range(num_folds)]
    
    return np.array(fold_indices)


def cross_validation(y, tx, fold_indices, current_fold, lambda_, gamma, max_iters):
    # Extract indices for the test set from the current fold.
    test_indices = fold_indices[current_fold]

    # Extract indices for the training set by excluding the current fold.
    train_indices = fold_indices[~(np.arange(fold_indices.shape[0]) == current_fold)].reshape(-1)

    # Partition the target data into training and test sets.
    y_test = y[test_indices]
    y_train = y[train_indices]
    
    # Partition the feature data into training and test sets.
    tx_test = tx[test_indices]
    tx_train = tx[train_indices]
    
    # Perform regularized ridge regression on the training set.
    w, train_loss = reg_logistic_regression(y_train, tx_train, lambda_, initial_w, max_iters, gamma)
    
    
    # Calculate the loss on the test set using the learned weights (without re-training).
    _, test_loss = calculate_loss(y_test, tx_test, w)
    
    return train_loss, test_loss, w


def cross_validation_loop(y, tx, k_fold, lambda_, seed, gamma, max_iters):
    # Generate the indices for k-fold cross-validation.
    k_indices = generate_k_fold_indices(y, k_fold, seed)

    # Lists to store the RMSE values for each fold.
    rmse_tr = []
    rmse_te = []

    # Loop over each fold.
    for k in range(k_fold):
        # Perform cross-validation for the current fold.
        loss_tr, loss_te, _ = cross_validation(y, tx, k_indices, k, lambda_, gamma, max_iters)
        
        # Convert losses to RMSE and append to the respective lists.
        rmse_tr.append(np.sqrt(2 * loss_tr))
        rmse_te.append(np.sqrt(2 * loss_te))
        
    # Compute the average RMSE values over all k-folds
    avg_rmse_tr = np.mean(rmse_tr)
    avg_rmse_te = np.mean(rmse_te)
    
    return avg_rmse_tr, avg_rmse_te


def best_lambda(y, tx, k_fold, lambdas, seed, gamma, max_iters):
    k_indices = generate_k_fold_indices(y, k_fold, seed)
    best_rmse = float('inf')
    best_lambda = None

    for lambda_ in lambdas:
        rmse_te_total = 0
        for k in range(k_fold):
            _, loss_te, _ = cross_validation(y, tx, k_indices, k, lambda_, gamma, max_iters)
            rmse_te_total += np.sqrt(2 * loss_te)
        
        rmse_te_avg = rmse_te_total / k_fold

        if rmse_te_avg < best_rmse:
            best_rmse = rmse_te_avg
            best_lambda = lambda_
            
    return best_lambda


def reg_logistic_regression_cross_validation(y, tx, k_folds, lambdas, gamma, max_iters):
    best_lambda_val = best_lambda(y, tx, k_folds, lambdas, seed, gamma, max_iters)
    w, loss = reg_logistic_regression(y, tx, best_lambda_val, initial_w, max_iters, gamma)
    return w, loss, best_lambda_val
   
