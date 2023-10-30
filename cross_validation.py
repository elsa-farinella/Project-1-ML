import numpy as np
from implementations import *
from additional_methods import *


def build_k_indices (y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation (y,tx, k_indices, k, lambda_ ):
    te_ind = k_indices[k]
    tr_ind = k_indices[~(np.arange(k_indices.shape[0])==k)]
    tr_ind = tr_ind.reshape(-1)
    y_te   = y[te_ind]
    y_tr   = y[tr_ind]
    tx_te  = tx[te_ind]
    tx_tr  = tx[tr_ind]
    w,loss_tr = ridge_regression(y_tr, tx_tr, lambda_)
    _,loss_te = ridge_regression(y_te,tx_te, lambda_)
    return loss_tr, loss_te,w


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
    
