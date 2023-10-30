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


def cross_validation (y,tx, fold_indices, current_fold, lambda_ ):
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
    w, train_loss = ridge_regression(target_train, data_train, regularization_param)
    
    # Calculate the loss on the test set using the learned weights (without re-training).
    _, test_loss = ridge_regression(target_test, data_test, regularization_param)
    
    return train_loss, test_loss, w


def cross_validation_loop(y, tx, k_fold, lambda_, degree, seed):
    """Compute the average training and test RMSE for a given lambda and degree."""
    k_indices = build_k_indices(y, k_fold, seed)
    rmse_tr = []
    rmse_te = []
    
    # Create polynomial features for the data
    tx_poly = build_poly(tx, degree)
    
    for k in range(k_fold):
        loss_tr, loss_te, _ = cross_validation(y, tx_poly, k_indices, k, lambda_)
        rmse_tr.append(np.sqrt(2 * loss_tr))
        rmse_te.append(np.sqrt(2 * loss_te))
    
    # Compute the average RMSE values over all k-folds
    avg_rmse_tr = np.mean(rmse_tr)
    avg_rmse_te = np.mean(rmse_te)
    
    return avg_rmse_tr, avg_rmse_te
    
def best_lambda_degree (y, tx,k_fold, lambdas, degrees,seed):
    k_indices = build_k_indices(y, k_fold, seed)
    best_lambdas = []
    best_rmses   = []
    #for each degree, save lambdas 
    for degree in degrees : 
        rmse_te = []
        for lambda_ in lambdas : 
            rmse_te1 = []
            for k in range(k_fold): 
                _, loss_te,_ = cross_validation (y, tx, k_indices, k, lambda_)
                rmse_te1.append(loss_te)
            rmse_te.append(np.mean(rmse_te1))
        
        indice_lambda = np.argmin(rmse_te)
        best_lambdas.append ( lambdas[indice_lambda])
        best_rmses.append( rmse_te[indice_lambda])
    
    indice_deg = np.argmin(best_rmses)
    return  best_lambdas[indice_deg],degrees[indice_deg]
    
def ridge_regression_cross_validation ( y,tx, k, lambdas, degrees) : 
    seed = 1
    k_indices = build_k_indices ( y, k, seed)
    lambda_, degree = best_lambda_degree(y,tx,k, lambdas, degrees,seed)
    tx = build_poly (tx, degree)
    w,loss = ridge_regression(y,tx,lambda_)
    return w,loss, degree
    
