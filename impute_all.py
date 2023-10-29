from data_imputation import *
import numpy as np


def main(train=True):
    # We store the paths of original dataset and labels
    if train:
        path_dataset = "/home/rfaro/anaconda3/EPFL_ML/Project1/dataset_to_release/x_train.csv"
        filename = "train_data_imputed.csv"
    else:
        path_dataset = "/home/rfaro/anaconda3/EPFL_ML/Project1/dataset_to_release/x_test.csv"
        filename = "test_data_imputed.csv"

    
    # We select the columns we want to keep
    data = np.genfromtxt(path_dataset,
                        delimiter=",",
                        skip_header=1,
                        usecols=[27, 31, 33, 38, 39, 40, 49, 51, 53, 61, 70, 88, 261, 266, 277, 278, 28, 29, 34, 35, 41, 43, 44, 45, 46, 47, 48,
                                66, 137, 145, 248, 1, 32, 255, 307]) 
                        #27, 31, 33, 38, 39, 40, 49, 51, 53, 61, 70, 88, 261, 266, 277, 278
                        #27, 35, 40, 49, 250, 28, 261, 266, 306, 259

    # Preprocessing of the data: this key step consists in replacing the missing values with -1, both Nan and numerical values, that 
    # do not have a meaning according the codebook.
    # Additionally, the data is scaled for uniformity, and categorical values such as 'No' and 'Yes' are encoded as '0' and '1', respectively.

    data[np.isnan(data)] = -1

    data[:, 0][data[:, 0] > 5] = -1
    data[:, 0][data[:, 0] >= 0] /= 5

    data[:, 1][data[:, 1] > 2] = -1
    data[:, 1][data[:, 1] == 2] = 0

    data[:, 2][data[:, 2] > 2] = -1
    data[:, 2][data[:, 2] == 2] = 0

    data[:, 3][data[:, 3] > 4] = -1
    data[:, 3][data[:, 3] >= 0] /= 4

    data[:, 4][data[:, 4] > 2] = -1
    data[:, 4][data[:, 4] == 2] = 0

    data[:, 5][data[:, 5] > 2] = -1
    data[:, 5][data[:, 5] == 2] = 0

    data[:, 6][data[:, 6] > 4] = -1
    data[:, 6][data[:, 6] == 2] = 0
    data[:, 6][data[:, 6] == 3] = 0
    data[:, 6][data[:, 6] == 4] = 2
    data[:, 6][data[:, 6] >= 0] /= 2

    data[:, 7][data[:, 7] == 2] = 0

    data[:, 8][data[:, 8] > 6] = -1
    data[:, 8][data[:, 8] >= 0] /= 6

    data[:, 9][data[:, 9] > 8] = -1
    data[:, 9][data[:, 9] >= 0] /= 8

    data[:, 10][data[:, 10] > 2] = -1
    data[:, 10][data[:, 10] == 2] = 0

    data[:, 11][data[:, 11] > 2] = -1
    data[:, 11][data[:, 11] == 2] = 0

    data[:, 12][data[:, 12] > 2] = -1
    data[:, 12][data[:, 12] == 1] = 0
    data[:, 12][data[:, 12] == 2] = 1

    data[:, 13][data[:, 13] > 2] = -1
    data[:, 13][data[:, 13] == 1] = 0
    data[:, 13][data[:, 13] == 2] = 1

    data[:, 14][data[:, 14] > 10] = 10
    data[:, 14][data[:, 14] >= 0] /= 10

    data[:, 15][data[:, 15] > 10] = 10
    data[:, 15][data[:, 15] >= 0] /= 10

    data[:, 16:18][data[:, 16:18] > 30] = -1
    data[:, 16:18][data[:, 16:18] >= 0] /= 30

    data[:, 18][data[:, 18] > 4] = -1
    data[:, 18][data[:, 18] >= 0] /= 4

    data[:, 19][data[:, 19] > 4] = -1
    data[:, 19][data[:, 19] == 4] = 1
    data[:, 19][data[:, 19] >= 2] = 0

    data[:, 20][data[:, 20] > 2] = -1
    data[:, 20][data[:, 20] == 2] = 0

    data[:, 21][data[:, 21] > 2] = -1
    data[:, 21][data[:, 21] == 2] = 0

    data[:, 22][data[:, 22] > 2] = -1
    data[:, 22][data[:, 22] == 2] = 0

    data[:, 23][data[:, 23] > 2] = -1
    data[:, 23][data[:, 23] == 2] = 0

    data[:, 24][data[:, 24] > 2] = -1
    data[:, 24][data[:, 24] == 2] = 0

    data[:, 25][data[:, 25] > 2] = -1
    data[:, 25][data[:, 25] == 2] = 0

    data[:, 26][data[:, 26] > 2] = -1
    data[:, 26][data[:, 26] == 2] = 0

    data[:, 27][data[:, 27] > 2] = -1
    data[:, 27][data[:, 27] == 2] = 0

    data[:, 28][data[:, 28] > 2] = -1
    data[:, 28][data[:, 28] == 2] = 0

    data[:, 29][data[:, 29] > 2] = -1
    data[:, 29][data[:, 29] == 2] = 0

    data[:, 30][data[:, 30] > 2] = -1
    data[:, 30][data[:, 30] == 2] = 0

    data[:, 31][data[:, 31] > 72] = -1
    data[:, 31][data[:, 31] >= 0] /= 72

    data[:, 32][data[:, 32] > 3] = -1
    data[:, 32][data[:, 32] == 3] = 0
    data[:, 32][data[:, 32] == 2] = 1 
    data[:, 32][data[:, 32] >= 0] /= 3

    data[:, 33][data[:, 33] > 4] = -1
    data[:, 33][data[:, 33] >= 0] /= 4

    data[:, 34][data[:, 34] > 2] = -1
    data[:, 34][data[:, 34] == 2] = 0

    # We define a vector containing the indices of columns for which we intend to compute the mean during data imputation. 
    continuous_columns = [0, 3, 8, 9, 14, 15, 16, 17, 31, 33]

    data = data_imputation(data, 100, continuous_columns)

    # Due to the extended duration of the imputation process —surpassing 45 minutes— it is prudent to store the imputed data in a CSV file.
    # This allows for immediate utilization without the need to await the completion of future data imputations.
    np.savetxt(filename, data, delimiter=",")





