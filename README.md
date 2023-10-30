# ML PROJECT 1 

# Team
- Team name: 
- Team member:
    - Elsa Farinella, SCIPER: 377583
    - Robin Faro, SCIPER:
    - Marco Scialanga, SCIPER: 369469

# Codebase
The project's codebase is structured across the following files:

**implementations.py**: This file contains all the methods we have been instructed to implement, supplemented by auxiliary functions. We defined *compute_MSE* and *compute_gradient* methods to calculate the gradient of the mean squared loss with the respect of the weights of the linear regressor. These helper functions are utilized within the first four main methods.

For the final two methods centered around logistic regression, we introduced:

- *sigmoid_function* to apply the sigmoid transformation to a given input.
- *compute_logistic_loss* to determine the Cross Entropy Loss.
- *compute_logistic_gradient* to compute the gradient of the Cross Entropy Loss concerning the regressor weights.

**additional_methods.py**: This file contains supplementary functions used during our experimentation:

- *build_model_data* adds a bias column to a specified feature matrix.
- *split_data* divides the dataset into training and validation subsets.
- *normalize_data* computes the normalization of data, thereby ensuring more consistent convergence.
- *polynomial_regression* offers an alternative regression technique for comparative analysis.

We've also integrated metric functions to compute F1 scores, model accuracy, and to plot confusion matrices based on model predictions.

**data_imputation.py**: This file contains a method designed for missing data imputation. The methodology resembles KNN imputation, where "nearness" is measured using the Euclidean distance between rows.

**impute_all.py**: This script produces a .csv file filled with imputed data. It utilizes the previously mentioned imputation method and also conducts essential data preprocessing. It's essential to note that all missing values, whether they are NaN or numerical, are set to -1 before the matrix undergoes the data imputation process.

**run.py**: This file allows the reproduction of the results obtained in AICrowd submission (ID submission: )

**DATASET**: Due to the lengthy execution times of data imputation, the imputed datasets (both training and testing) are stored in a .zip folder, available in .csv format.

# Instructions on how to run 

## Requirements 
- Python==
- Numpy==
- Matplotlib==


