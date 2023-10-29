import numpy as np
from implementations import *
import matplotlib.pyplot as plt


### helper function to add the bias column in the feature matrix ###

def build_model_data(data):
    """Form (y,tX) to get regression data in matrix form."""
    num_samples = data.shape[0]
    tx = np.c_[np.ones(num_samples), data]
    return tx

### split data function ###

def split_data(x, y, ratio, seed=1):
    np.random.seed(seed)
    N = len(x)
    N_tr = int(N * ratio)
    N_te = N - N_tr
    idx = np.random.permutation(N)
    idx_tr = idx[:N_tr]
    idx_te = idx[N_tr:]
    x_tr = x[idx_tr]
    x_te = x[idx_te]
    y_tr = y[idx_tr]
    y_te = y[idx_te]

    train_data = np.c_[x_tr, y_tr]
    test_data = np.c_[x_te, y_te]
    return train_data, test_data


### data normalization ###

def normalize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    epsilon = 1e-8      # small number to avoid division by zero
    std[std < epsilon] = epsilon
    data = (data - mean) / std
    return data

#### polynomial regression ####

def build_poly(x, degree):
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    poly_tx = np.ones((x.shape[0], 1))
    for i in range(1, degree+1):
        poly_tx = np.c_[poly_tx, np.power(x, i)]
    return poly_tx

def polynomial_regression(y, tx, degree, max_iters, gamma, print_=True):
    poly_tx = build_poly(tx, degree)
    w = np.zeros(poly_tx.shape[1])
    for n_iter in range(max_iters):
        loss = compute_MSE(y, poly_tx, w)
        gradient = compute_gradient(y, poly_tx, w)
        w = w - gamma*gradient

        if(print_):
            print(
                "SGD Polynomial Regression iter. {bi}/{ti}: loss={l}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss
                )
            )
        

    return w, loss


### Additional metrics functions ###

def compute_accuracy(y, tx, w, threshold=0.5):
    pred_y = np.dot(tx, w)
    pred_y[pred_y <= threshold] = 0
    pred_y[pred_y > threshold] = 1
    return np.sum(pred_y == y) / len(y)

def confusion_matrix(y, tx, w, threshold=0.5):
    pred_y = np.dot(tx, w)
    pred_y[pred_y <= threshold] = 0
    pred_y[pred_y > threshold] = 1
    tp = np.sum(pred_y[y == 1] == 1)
    tn = np.sum(pred_y[y == 0] == 0)
    fp = np.sum(pred_y[y == 0] == 1)
    fn = np.sum(pred_y[y == 1] == 0)
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("Blues") 
    tot_pos = tp + fn
    tot_neg = fp + tn
    im = ax.imshow(np.array([[tp, fp], [fn, tn]]),  cmap=cmap)
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(["Positive", "Negative"])
    ax.set_yticklabels(["Positive", "Negative"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, np.array([[tp/tot_pos, fp/tot_neg], [fn/tot_pos, tn/tot_neg]])[i][j], ha="center", va="center")
    fig.tight_layout()
    plt.show()


def compute_f1(y, tx, w, threshold=0.5):
    pred_y = np.dot(tx, w)
    pred_y[pred_y <= threshold] = 0
    pred_y[pred_y > threshold] = 1
    tp = np.sum(pred_y[y == 1] == 1)
    tn = np.sum(pred_y[y == 0] == 0)
    fp = np.sum(pred_y[y == 0] == 1)
    fn = np.sum(pred_y[y == 1] == 0)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2*(precision * recall) / (precision + recall)


def compute_f1_logistic(y, tx, w, threshold=0.5):
    pred_y = sigmoid(np.dot(tx, w))
    pred_y[pred_y <= threshold] = 0
    pred_y[pred_y > threshold] = 1
    tp = np.sum(pred_y[y == 1] == 1)
    tn = np.sum(pred_y[y == 0] == 0)
    fp = np.sum(pred_y[y == 0] == 1)
    fn = np.sum(pred_y[y == 1] == 0)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2*(precision * recall) / (precision + recall)
    
