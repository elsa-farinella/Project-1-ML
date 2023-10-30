from implementations import *
from additional_methods import *
import numpy as np


def main():
    # First of all we define two variables in which we store the paths of the dataset and the labels
    # The data are stored in a csv file we created through the script data_imputation.py. Read the README.md file for more information
    path_dataset = "DATASET/train_data_imputed.csv"
    path_labels = "DATASET/y_train.csv"

    data = np.genfromtxt(path_dataset, delimiter=",")

    y = np.genfromtxt(
        path_labels,
        delimiter=",",
        skip_header=1,
        usecols=[1]
    )

    # For an easier implementation of our functions, we changed the label -1 to 0
    y[y== -1] = 0

    # Then we split the data into training and a local validation set
    train_data, val_data = split_data(data, y, 0.95, seed=13)

    # We split the data into features and labels, since they were put together inside the previous function in order not to lose the correspondence between them
    val_x = val_data[:, 0:-1]
    val_y = val_data[:, -1]

    # Before processing the training data similarly, we remove some rows to ensure a balanced dataset for model training.
    keep_percentage = 0.36
    np.random.seed(13)
    # We compute the number of zero rows to keep, adn we save the non-zero ones
    zero_rows = train_data[train_data[:, -1] == -0]
    non_zero_rows = train_data[train_data[:, -1] != -0]
    num_rows_to_keep = int(len(zero_rows) * keep_percentage)
    # We choose randomly the zero-rows to keep
    selected_rows = np.random.choice(zero_rows.shape[0], num_rows_to_keep, replace=False)
    # In the end we join the selected rows with the non-zero rows
    filtered_data = zero_rows[selected_rows]
    filtered_data = np.concatenate((filtered_data, non_zero_rows), axis=0)

    # Now we can split features and labels also in train set
    train_x = filtered_data[:, :-1]
    train_y = filtered_data[:, -1]

    # We normalize the sets of data
    train_x_normalized = normalize_data(train_x)
    val_x_normalized = normalize_data(val_x)

    # We use a logistic regression model on the normalized data
    train_tx = build_model_data(train_x_normalized)
    val_tx = build_model_data(val_x_normalized)
    max_iters = 1000
    gamma = 1
    initial_w = np.full(train_tx.shape[1], -1)
    w, loss = logistic_regression(train_y, train_tx, initial_w, max_iters, gamma, False)
    # We compute some metrics on the validation set to evaluate our model
    f1 = compute_f1_logistic(val_y, val_tx, w)
    print("F1 score: ", f1)
    accuracy = compute_accuracy(val_y, val_tx, w)
    print("Accuracy: ", accuracy)
    confusion_matrix(val_y, val_tx, w)


if __name__ == "__main__":
    main()

