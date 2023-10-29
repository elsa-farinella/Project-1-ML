import numpy as np

def euclidean_distances(data, point):
    return np.linalg.norm(data - point, axis=1)


def data_imputation(data, max_neighbors, mean_columns, distance_metric=euclidean_distances, max_samples=5000):
    # Ensure data is a numpy array and create a copy of it
    data = np.array(data) 
    imputed_data = data.copy()

    # Identify the indices of rows with missing values to ensure data imputation is applied exclusively to those specific entries.
    missing_rows = np.where((data[:, :] == -1).any(axis=1))[0]

    for row_idx in missing_rows:
        # Identify the indices of rows without '-1' values to use them for imputing missing data.
        label_indices = np.where((data[:, :] != -1).all(axis=1))[0]
        # If there are more than max_samples neighbors, select a random subset of them to reduce computation time
        if len(label_indices) > max_samples:
            label_indices = np.random.choice(label_indices, max_samples, replace=False)

        missing_values = (data[row_idx, :] == -1)  # Identify indexes of missing values in the current row

        # We establish a variable to identify 'continuous' columns, which will be imputed using the mean rather than the mode.
        mean_to_be_done = np.isin(np.arange(data.shape[1]), mean_columns)

        # Calculate distances between the current row and all neighbors
        distances = distance_metric(data[label_indices, :], data[row_idx, :])

        # Sort the neighbors by distance and select the first 'max_neighbors'
        sorted_neighbors = label_indices[np.argsort(distances)]
        valid_neighbors = sorted_neighbors[:max_neighbors]

        neighbor_values = data[valid_neighbors][:, :]
        imputed_values = imputed_data[row_idx, :]

        # For columns with discrete data, compute the mode of neighboring values to impute each missing entry
        if np.any(~mean_to_be_done & missing_values):
            for col_idx in np.where(~mean_to_be_done & missing_values)[0]:
                col_values = neighbor_values[:, col_idx]
                unique_values, counts = np.unique(col_values, return_counts=True)
                mode_idx = np.argmax(counts)
                imputed_values[col_idx] = unique_values[mode_idx]

        # For columns with continous data, compute the mean of neighboring values to impute each missing entry
        if np.any(mean_to_be_done & missing_values):
            for col_idx in np.where(mean_to_be_done & missing_values)[0]:
                col_values = neighbor_values[:, col_idx]
                imputed_values[col_idx] = np.mean(col_values)

        imputed_data[row_idx, :] = imputed_values

    return imputed_data



    
