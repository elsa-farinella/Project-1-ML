import numpy as np
from implementations import *
import matplotlib.pyplot as plt


# Helper function to add the bias column in the feature matrix

def build_model_data(data):
    """Form (y,tX) to get regression data in matrix form."""
    num_samples = data.shape[0]
    tx = np.c_[np.ones(num_samples), data]
    return tx

# Helper function to load the test set

def load_test():
    path_test = "DATASET/test_data_imputed.csv"
    test_data = np.genfromtxt(path_test, delimiter=",")
    return test_data


# Split data function

def split_data(x, y, ratio, seed=1):
    np.random.seed(seed)
    N = len(x)

    # Calculate the number of training samples based on the ratio and the number of testing samples
    N_tr = int(N * ratio)
    N_te = N - N_tr

    idx = np.random.permutation(N)

    # Split the indices for training and testing data
    idx_tr = idx[:N_tr]
    idx_te = idx[N_tr:]

    # Use the indices to obtain the corresponding data
    x_tr = x[idx_tr]
    x_te = x[idx_te]
    y_tr = y[idx_tr]
    y_te = y[idx_te]

     # Concatenate the features and labels for both training and testing data
    train_data = np.c_[x_tr, y_tr]
    test_data = np.c_[x_te, y_te]
    return train_data, test_data


# Data normalization

def normalize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    epsilon = 1e-8      # Small number to avoid division by zero
    std[std < epsilon] = epsilon
    data = (data - mean) / std
    return data


# Oversampling the minority class

def oversample_positive_labels(df, oversampling_ratio):
  # Extract the target column as a NumPy array
  target_column = df[:,-1]

  # Find the indices of positive samples
  positive_indices = np.where(target_column == 1)[0]

  # Determine the number of positive samples
  num_positive_samples = len(positive_indices)

  if num_positive_samples == 0:
    # No positive samples, return the original DataFrame
    return df

  # Calculate the number of samples needed to reach the desired oversampling ratio
  oversample_count = int((oversampling_ratio - 1.0) * num_positive_samples)

  # Randomly choose indices to oversample with replacement
  oversample_indices = np.random.choice(positive_indices, oversample_count, replace=True)

  # Extract the rows to oversample from the original DataFrame
  oversampled_rows = df[oversample_indices,:]

  # Stack the oversampled rows on top of the original array
  oversampled_data = np.vstack([df, oversampled_rows])

  return oversampled_data


# Polynomial regression

def build_poly(x, degree):
    # This function should return the matrix formed by applying the polynomial basis to the input data
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


# Additional metrics functions 

# We introduced the 'threshold' parameter to the functions in response to the data's notable skew towards negative values. 
# This adjustment aims to promote a more balanced prediction from the model. 
# However, adjusting the threshold did not provide significant improvements, so it was ultimately left at the conventional value of 0.5.

def compute_accuracy(y, tx, w, threshold=0.5):
    # Calculate the predictions based on weights and features matrix
    pred_y = np.dot(tx, w)

    # Classify the predictions based on the threshold value
    pred_y[pred_y <= threshold] = 0
    pred_y[pred_y > threshold] = 1
    
    return np.sum(pred_y == y) / len(y)

def confusion_matrix(y, tx, w, threshold=0.5):
    # Calculate the predictions based on weights and features matrix
    pred_y = np.dot(tx, w)

    # Classify the predictions based on the threshold value
    pred_y[pred_y <= threshold] = 0
    pred_y[pred_y > threshold] = 1

    # Calculate confusion matrix components
    tp = np.sum(pred_y[y == 1] == 1) # Count the true positive predictions
    tn = np.sum(pred_y[y == 0] == 0) # Count the true negative predictions
    fp = np.sum(pred_y[y == 0] == 1) # Count the false positive predictions
    fn = np.sum(pred_y[y == 1] == 0) # Count the false negative predictions 

    # Prepare the visualization
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("Blues") 

    # Calculate total positives and negatives for normalization
    tot_pos = tp + fn
    tot_neg = fp + tn
    im = ax.imshow(np.array([[tp, fp], [fn, tn]]),  cmap=cmap)

    # Setting tick labels for the axes and the title for the plot
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(["Positive", "Negative"])
    ax.set_yticklabels(["Positive", "Negative"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    # Display the normalized values as text inside the heatmap
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, np.array([[tp/tot_pos, fp/tot_neg], [fn/tot_pos, tn/tot_neg]])[i][j], ha="center", va="center")
    
    # Adjust layout for better visualization and display the plot
    fig.tight_layout()
    plt.show()


def compute_f1(y, tx, w, threshold=0.5):
    # Calculate the predictions based on weights and features matrix
    pred_y = np.dot(tx, w)

    # Classify the predictions based on the threshold value
    pred_y[pred_y <= threshold] = 0
    pred_y[pred_y > threshold] = 1

    # Calculate confusion matrix components
    tp = np.sum(pred_y[y == 1] == 1) # True positive
    tn = np.sum(pred_y[y == 0] == 0) # True negative
    fp = np.sum(pred_y[y == 0] == 1) # False positive
    fn = np.sum(pred_y[y == 1] == 0) # False negative

    # Calculate precision and recall
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    # Compute and return the F1 score
    return 2*(precision * recall) / (precision + recall)


def compute_f1_logistic(y, tx, w, threshold=0.5):
    # Calculate the logistic predictions using sigmoid function
    pred_y = sigmoid(np.dot(tx, w))

    # Classify the predictions based on the threshold value
    pred_y[pred_y <= threshold] = 0
    pred_y[pred_y > threshold] = 1

    # Calculate confusion matrix components
    tp = np.sum(pred_y[y == 1] == 1)
    tn = np.sum(pred_y[y == 0] == 0)
    fp = np.sum(pred_y[y == 0] == 1)
    fn = np.sum(pred_y[y == 1] == 0)

    # Calculate precision and recall
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    # Compute and return the F1 score
    return 2*(precision * recall) / (precision + recall)
    
