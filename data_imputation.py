import numpy as np

def euclidean_distances(data, point):
    return np.linalg.norm(data - point, axis=1)

def data_imputation(data, max_neighbors, mean_columns, distance_metric=euclidean_distances, max_samples=5000):
    data = np.array(data)  # Ensure data is a NumPy array
    imputed_data = data.copy()

    # calcolo le righe che presentano valori mancanti così da iterare solo su quelle
    missing_rows = np.where((data[:, :-1] == -1).any(axis=1))[0]

    for row_idx in missing_rows:

        # Get indices of rows that don't have -1 values
        label_indices = np.where((data[:, :-1] != -1).all(axis=1))[0]


        # If there are more than max_samples neighbors, select a random subset of them
        if len(label_indices) > max_samples:
            label_indices = np.random.choice(label_indices, max_samples, replace=False)

        missing_values = (data[row_idx, :-1] == -1)  # Identify missing values in the current row

        # Determine if the column is discrete
        mean_to_be_done = np.isin(np.arange(data.shape[1] - 1), mean_columns)

        # Calculate distances between the current row and all neighbors
        distances = distance_metric(data[label_indices, :-1], data[row_idx, :-1])

        # Sort the neighbors by distance and select the top max_neighbors
        sorted_neighbors = label_indices[np.argsort(distances)]
        valid_neighbors = sorted_neighbors[:max_neighbors]

        neighbor_values = data[valid_neighbors][:, :-1]
        imputed_values = imputed_data[row_idx, :-1]

        # For discrete columns, calculate the mode of neighbor values for each missing value
        if np.any(~mean_to_be_done & missing_values):
            for col_idx in np.where(~mean_to_be_done & missing_values)[0]:
                col_values = neighbor_values[:, col_idx]
                unique_values, counts = np.unique(col_values, return_counts=True)
                mode_idx = np.argmax(counts)
                imputed_values[col_idx] = unique_values[mode_idx]

        # For continuous columns, calculate the mean of neighbor values for each missing value
        if np.any(mean_to_be_done & missing_values):
            for col_idx in np.where(mean_to_be_done & missing_values)[0]:
                col_values = neighbor_values[:, col_idx]
                imputed_values[col_idx] = np.mean(col_values)

        imputed_data[row_idx, :-1] = imputed_values

    return imputed_data



def data_test_imputation(data, max_neighbors, discrete_columns, distance_metric=euclidean_distances, max_samples=1000):
    data = np.array(data)  # Ensure data is a NumPy array
    imputed_data = data.copy()

    # calcolo le righe che presentano valori mancanti così da iterare solo su quelle
    missing_rows = np.where((data[:, :] == -1).any(axis=1))[0]

    for row_idx in missing_rows:

        # Get indices of rows with the same label who don't have -1 values
        label_indices = np.where((data[:, :] != -1).all(axis=1))[0]


        # If there are more than max_samples neighbors, select a random subset of them
        if len(label_indices) > max_samples:
            label_indices = np.random.choice(label_indices, max_samples, replace=False)

        missing_values = (data[row_idx, :] == -1)  # Identify missing values in the current row

        # Determine if the column is discrete
        is_discrete = np.isin(np.arange(data.shape[1]), discrete_columns)

        # Calculate distances between the current row and all neighbors
        distances = distance_metric(data[label_indices, :], data[row_idx, :])

        # Sort the neighbors by distance and select the top max_neighbors
        sorted_neighbors = label_indices[np.argsort(distances)]
        valid_neighbors = sorted_neighbors[:max_neighbors]

        neighbor_values = data[valid_neighbors][:, :]
        imputed_values = imputed_data[row_idx, :]

        # For discrete columns, calculate the mode of neighbor values for each missing value
        if np.any(is_discrete & missing_values):
            for col_idx in np.where(is_discrete & missing_values)[0]:
                col_values = neighbor_values[:, col_idx]
                unique_values, counts = np.unique(col_values, return_counts=True)
                mode_idx = np.argmax(counts)
                imputed_values[col_idx] = unique_values[mode_idx]

        # For continuous columns, calculate the mean of neighbor values for each missing value
        if np.any(~is_discrete & missing_values):
            for col_idx in np.where(~is_discrete & missing_values)[0]:
                col_values = neighbor_values[:, col_idx]
                imputed_values[col_idx] = np.mean(col_values)

        imputed_data[row_idx, :] = imputed_values

    return imputed_data




# usage example

# data = np.array([[1, 2, 3, -1, 0],
#                  [1, -1, 3, 4, 1],
#                  [1, 2, -1, 4, 1],
#                  [2, -1, 5, 6, 0],
#                  [2, 3, -1, 5, 0]])

# proximity_radius = 100 # Specifica il raggio di prossimità
# discrete_columns = {2}  # Specifica gli indici delle colonne discrete

# imputed_data = data_imputation(data, proximity_radius, discrete_columns, distance_metric=euclidean_distance, max_samples=1000)

# print("Dati originali:")
# print(data)
# print("\nDati imputati:")
# print(imputed_data)
