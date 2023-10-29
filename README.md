# ML PROJECT 1 

The code is organized in the following files:

**implementations.py** This file contains all the methods that we were asked to implement. Clearly together with some other functions useful for the computation done in the main functions. Specifically, we defined a *compute_MSE* and a *compute_gradient* method which calculates the gradient of the mean squared loss with the respect to the weights of the linear regressor. We used this two helper function inside the first 4 assigned methods. For the last two, which were about the logistic regression, we also defined the *sigmoid* function, which as the name suggests apply the sigmoid function to a given input, a *compute_logistic_loss* that instead computes the so called Cross Entropy Loss, and the *compute_logistic_gradient* method which computes the gradient of the CE Loss with the respect to the weights.

**additional_methods.py** This file contains some useful functions we used during our experiments. First of all we wrote a *build_model_data* method which adds the bias column in a given feature matrix. Then we implemented *split_data*, a fucntion that splits the data in a train and validation set. We wrote *normalize_data* in order to normalize the date and achieve a smoother covenrgence. There is also an implementation of polynomial_regression, whihch we wanted to exploit for comparison, and, in the end, there also are some metric functions, in order to compute F1 score and accuracy of a given model and also plot a confusion matrix based on the model predictions.

**data_imputation.py** This file contains just a function to impute the value of the missing data. This is a sort of KNN imputation, since it performs the imputation of a certain number of nearest sample, where the concept of near is realted to the eucildean distance between rows. 

**impute_all.py** This file generates a csv file that containes the imputed data. It obviously relies on the above-defined method, but it also performs some data preprocessing, that is important since every missing values, both Nan and numerical, has to be put to -1 before that the matrix is sent to the data_imputation fucntion.

**run.py** This file allows the reproduction of the results obtained in AICrowd submission

**DATASET** As said before, the data have been imputed and it took a bit, so there is a .zip folder which contains the imputed train and test set saved as .cvs files.


--------------------
The project's codebase is structured across the following files:

**implementations.py**: This file contains all the methods required for implementation, supplemented by auxiliary functions. We defined *compute_MSE* and *compute_gradient* methods to calculate the gradient of the mean squared loss with the respect of the weights of the linear regressor. These helper functions are utilized within the first four main methods.

For the final two methods centered around logistic regression, we introduced:

- sigmoid_function to apply the sigmoid transformation to a given input.
- compute_logistic_loss to determine the Cross Entropy Loss.
- compute_logistic_gradient to compute the gradient of the Cross Entropy Loss concerning the regressor weights.

**additional_methods.py**: This file contains supplementary functions used during our experimentation:

- *build_model_data* adds a bias column to a specified feature matrix.
- *split_data* divides the dataset into training and validation subsets.
- *normalize_data* aids in data normalization for more consistent convergence.
- *polynomial_regression* offers an alternative regression technique for comparative analysis.

We've also integrated metric functions to compute F1 scores, model accuracy, and to plot confusion matrices based on model predictions.

**data_imputation.py**: This file contains a method designed for missing data imputation. The methodology resembles KNN imputation, where "nearness" is measured using the Euclidean distance between rows.

**impute_all.py**: This script produces a .csv file filled with imputed data. It utilizes the previously mentioned imputation method and also conducts essential data preprocessing. It's essential to note that all missing values, whether they are NaN or numerical, are set to -1 before the matrix undergoes the data imputation process.

**run.py**: This file allows the reproduction of the results obtained in AICrowd submission

**DATASET**: Due to the lengthy execution times of data imputation, the imputed datasets (both training and testing) are stored in a .zip folder, available in .csv format.
