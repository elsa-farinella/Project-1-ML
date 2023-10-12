from implementations import *
from load_data import *
import numpy as np
import datetime

path_dataset = "Project1/dataset/x_train.csv"
path_labels = "Project1/dataset/y_train.csv"
# data = np.genfromtxt(path_dataset,
#             delimiter=",",
#             skip_header=1,
#             usecols=[26, 34, 39, 48, 249, 253],
#             converters= {26: lambda x: 9 if int(x) > 5 else int(x),
#                         34: lambda x: 0 if int(x) == 3 else 1 if int(x) == 1 else 9,
#                         39: lambda x: 0 if int(x) == 2 else 1 if int(x) == 1 else 9,
#                         249: lambda x: 0 if int(x) < 6 else 1,
#                         253: lambda x: 0 if int(x) < 25 else 1,
#                         48: lambda x: 0 if int(x) == 3 else 1 if int(x) == 1 else 9,
#                         }
# )

import numpy as np

path_dataset = "Project1/dataset/x_train.csv"
path_labels = "Project1/dataset/y_train.csv"
data = np.genfromtxt(path_dataset,
            delimiter=",",
            skip_header=1,
            usecols=[26, 34, 39, 48, 249, 253],
)

y = np.genfromtxt(
    path_labels,
    delimiter=",",
    skip_header=1,
    usecols=[0]
)

data = np.c_[data, y]

# Rimuovi righe con valori NaN
data = data[~np.isnan(data).any(axis=1)]


data[:, 0][data[:, 0] > 5] = 9

data[:, 1][data[:, 1] > 3] = 9
data[:, 1][data[:, 1] == 3] = 0
data[:, 1][data[:, 1] == 2] = 9

data[:, 2][data[:, 2] > 2] = 9
data[:, 2][data[:, 2] == 2] = 0

data[:, 3][data[:, 3] > 3] = 9
data[:, 3][data[:, 3] == 3] = 0
data[:, 3][data[:, 3] == 2] = 9

data[:, 4][data[:, 4] < 6] = 0
data[:, 4][data[:, 4] >= 6] = 1

data[:, 5][data[:, 5] < 25] = 0 
data[:, 5][data[:, 5] >= 25] = 1

y = np.genfromtxt(
    path_labels,
    delimiter=",",
    skip_header=1,
    usecols=[0]
)

# rimuovere tutte le righe in cui un valore è maggiore di 5
data = data[np.where(data[:, 0] <= 5)]

import numpy as np

# Calcola la percentuale di righe da mantenere
percentage_to_keep = 0.1  # 10% da mantenere

# Trova le righe in cui il valore dell'ultima colonna è diverso da 0
zero_rows = data[data[:, 6] == 0]
non_zero_rows = data[data[:, 6] != 0]
# Calcola il numero di righe da mantenere
num_rows_to_keep = int(len(zero_rows) * percentage_to_keep)

# Seleziona casualmente le righe da mantenere
selected_rows = np.random.choice(zero_rows.shape[0], num_rows_to_keep, replace=False)

# Estrai le righe selezionate dall'array
filtered_data = zero_rows[selected_rows]
filtered_data = np.concatenate((filtered_data, non_zero_rows), axis=0)

# filtered_data ora contiene il 10% delle righe in cui il valore dell'ultima colonna è diverso da 0

