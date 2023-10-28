import numpy as np
from implementations import *
import matplotlib.pyplot as plt

#### polynomial regression ####

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
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



def confusion_matrix(y, tx, w, threshold=0.5):
    pred_y = np.dot(tx, w)
    pred_y[pred_y <= threshold] = 0
    pred_y[pred_y > threshold] = 1
    tp = np.sum(pred_y[y == 1] == 1)
    tn = np.sum(pred_y[y == 0] == 0)
    fp = np.sum(pred_y[y == 0] == 1)
    fn = np.sum(pred_y[y == 1] == 0)
    # plot confusion matrix
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("Blues")  # Puoi scegliere una colormap a tuo piacimento

    # calcolo il totale dei postivi e dei negativi
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
    