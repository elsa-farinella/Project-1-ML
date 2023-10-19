import numpy as np

def load_data():
    """Load data and convert it to the metric system."""
    path_dataset = "Project1/dataset_to_release/x_train.csv"
    path_labels = "Project1/dataset_to_release/y_train.csv"
    data = np.genfromtxt(path_dataset,
                     delimiter=",",
                     skip_header=1,
                     usecols=[27, 35, 40, 49, 250, 28]) 
    
    
    #  stroke39 diabete48 249age 
                    # converters= {26: lambda x: 9 if x > 5 else x, # GHEALTH
                    #              34: lambda x: 0 if x == 3 else 1 if x == 1 else 9, # Blood pressure
                    #              39: lambda x: 0 if x == 2 else 1 if x == 1 else 9, # stroke
                    #              249: lambda x: 0 if x < 6 else 1 , # age
                    #              253: lambda x: 0 if x < 25 else 1, # BMI
                    #              48: lambda x: 0 if x == 3 else 1 if x == 1 else 9, # diabete
                    #              }
                    # )
    
    y = np.genfromtxt(
        path_labels,
        delimiter=",",
        skip_header=1,
        usecols=[1]
    )

    data = np.c_[data, y]

    # Rimuovi righe con valori NaN
    #data = data[~np.isnan(data).any(axis=1)]

    #proviamo a sostituire i valori nan con -1
    data[np.isnan(data)] = -1

    data[:, 0][data[:, 0] > 5] = -1
    #normalize data
    data[:, 0][data[:, 0] >= 0] /= 5

    data[:, 1][data[:, 1] > 3] = -1
    data[:, 1][data[:, 1] == 3] = 0
    data[:, 1][data[:, 1] == 2] = -1

    data[:, 2][data[:, 2] > 2] = -1
    data[:, 2][data[:, 2] == 2] = 0

    data[:, 3][data[:, 3] > 3] = -1
    data[:, 3][data[:, 3] == 3] = 0
    data[:, 3][data[:, 3] == 2] = 1

    data[:, 4][data[:, 4] < 6] = 0
    data[:, 4][data[:, 4] >= 6] = 1

    data[:, 5][data[:, 5] == 88] = 0
    data[:, 5][data[:, 5] >= 77] = -1
    #normalize data
    data[:, 5][data[:, 5] >= 0] /= 30



    # Calcola la percentuale di righe da mantenere
    percentage_to_keep = 0.1  # 10% da mantenere

    # Trova le righe in cui il valore dell'ultima colonna è diverso da 0
    zero_rows = data[data[:, -1] == -1]
    non_zero_rows = data[data[:, -1] != -1]
    # Calcola il numero di righe da mantenere
    num_rows_to_keep = int(len(zero_rows) * percentage_to_keep)

    # Seleziona casualmente le righe da mantenere
    selected_rows = np.random.choice(zero_rows.shape[0], num_rows_to_keep, replace=False)

    # Estrai le righe selezionate dall'array
    filtered_data = zero_rows[selected_rows]
    filtered_data = np.concatenate((filtered_data, non_zero_rows), axis=0)

    x = filtered_data[:, 0:6]
    y = filtered_data[:, 6]
    return x, y

def build_model_data(data):
    """Form (y,tX) to get regression data in matrix form."""
    num_samples = data.shape[0]
    tx = np.c_[np.ones(num_samples), data]
    return tx



def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing. If ratio times the number of samples is not round
    you can use np.floor. Also check the documentation for np.random.permutation,
    it could be useful.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer.

    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.

    >>> split_data(np.arange(13), np.arange(13), 0.8, 1)
    (array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]), array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]))
    """
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    # ***************************************************
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
    return x_tr, x_te, y_tr, y_te


def load_test_data(): 
    """Load data and convert it to the metric system."""
    path_dataset = "Project1/dataset_to_release/x_test.csv"
    data = np.genfromtxt(path_dataset,
                     delimiter=",",
                     skip_header=1,
                     usecols=[27, 35, 40, 49, 250, 28]) 
    

    #proviamo a sostituire i valori nan con -1
    data[np.isnan(data)] = -1

    data[:, 0][data[:, 0] > 5] = -1
    data[:, 0][data[:, 0] >= 0] /= 5

    data[:, 1][data[:, 1] > 3] = -1
    data[:, 1][data[:, 1] == 3] = 0
    data[:, 1][data[:, 1] == 2] = -1

    data[:, 2][data[:, 2] > 2] = -1
    data[:, 2][data[:, 2] == 2] = 0

    data[:, 3][data[:, 3] > 3] = -1
    data[:, 3][data[:, 3] == 3] = 0
    data[:, 3][data[:, 3] == 2] = 1

    data[:, 4][data[:, 4] < 6] = 0
    data[:, 4][data[:, 4] >= 6] = 1

    data[:, 5][data[:, 5] == 88] = 0
    data[:, 5][data[:, 5] >= 77] = -1
    data[:, 5][data[:, 5] >= 0] /= 30



    return data
