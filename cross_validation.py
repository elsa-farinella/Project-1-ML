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


def cross_validation (y,tx, fold_indices, current_fold, lambda_):
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
    
    # Perform ridge regression on the training set.
    w, train_loss = ridge_regression(y_train, tx_train, lambda_)
    
    # Calculate the loss on the test set using the learned weights (without re-training).
    _, test_loss = ridge_regression(y_test, tx_test, lambda_)
    
    return train_loss, test_loss, w


def cross_validation_loop(y, tx, k_fold, lambda_, degree, seed):
    # Generate the indices for k-fold cross-validation.
    k_indices = generate_k_fold_indices(y, k_fold, seed)

    # Lists to store the RMSE values for each fold.
    rmse_tr = []
    rmse_te = []

    # Perform polynomial expansion on the feature matrix.
    tx_poly = build_poly(tx, degree)
    
    # Loop over each fold.
    for k in range(k_fold):
        # Perform cross-validation for the current fold.
        loss_tr, loss_te, _ = cross_validation(y, tx_poly, fold_indices, current_fold, lambda_)
        
        # Convert losses to RMSE and append to the respective lists.
        rmse_tr.append(np.sqrt(2 * loss_tr))
        rmse_te.append(np.sqrt(2 * loss_te))
        
    # Compute the average RMSE values over all k-folds
    avg_rmse_tr = np.mean(rmse_tr)
    avg_rmse_te = np.mean(rmse_te)
    
    return avg_rmse_tr, avg_rmse_te


def best_lambda_degree (y, tx, k_fold, lambdas, degree_values, seed):
    k_indices = generate_k_fold_indices(y, k_fold, seed)
    best_lambdas = [] # List to store the best lambda for each degree.
    best_rmse = []    # List to store the RMSE associated with the best lambda for each degree.

    # For each polynomial degree, find the best lambda.
    for degree in degree_values : 
        rmse_te = []

        for lambda_ in lambdas : 
            rmse_te1 = []
            for k in range(k_fold): 
                _, loss_te,_ = cross_validation (y, tx, fold_indices, current_fold, lambda_)
                rmse_te1.append(loss_te)
                
            # Compute average RMSE for the current lambda across all folds.
            rmse_te.append(np.mean(rmse_te1))

        # Find the lambda that gave the lowest RMSE for the current degree.
        index_lambda = np.argmin(rmse_te)
        best_lambdas.append (lambdas[index_lambda])
        best_rmses.append( rmse_te[index_lambda])

    # Find the degree (and corresponding lambda) that produced the lowest RMSE.
    index_deg = np.argmin(best_rmses)
    return  best_lambdas[index_deg],degrees[index_deg]


def ridge_regression_cross_validation (y, tx, k, lambdas, degrees) : 
    seed = 1
    k_indices = generate_k_fold_indices (y, k, seed)
    lambda_, degree = best_lambda_degree(y,tx,k, lambdas, degrees,seed)
    tx = build_poly (tx, degree)
    w,loss = ridge_regression(y,tx,lambda_)
    return w,loss, degree
    
