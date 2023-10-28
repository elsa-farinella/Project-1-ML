# ML PROJECT 1 

The code is organized in the following files:

**implementations.py** This file contains all the methods that we were asked to implement. Clearly together with some other functions useful for the computation done in the main functions. Specifically, we defined a *compute_MSE* and a *compute_gradient* method which calculates the gradient of the mean squared loss with the respect to the weights of the linear regressor. We used this two helper function inside the first 4 assigned methods. For the last two, which were about the logistic regression, we also defined the *sigmoid* function, which as the name suggests apply the sigmoid function to a given input, a *compute_logistic_loss* that instead computes the so called Cross Entropy Loss, and the *compute_logistic_gradient* method which computes the gradient of the CE Loss with the respect to the weights.

**additional_methods.py** This file contains some useful functions we used during our experiments. First of all we wrote a *build_model_data* method which adds the bias column in a given feature matrix. Then we implemented *split_data*, a fucntion that splits the data in a train and validation set. We wrote *normalize_data* in order to normalize the date and achieve a smoother covenrgence. There is also an implementation of polynomial_regression, whihch we wanted to exploit for comparison, and, in the end, there also are some metric functions, in order to compute F1 score and accuracy of a given model and also plot a confusion matrix based on the model predictions.

**data_imputation.py** This file contains just a function to impute the value of the missing data. This is a sort of KNN imputation, since it performs the imputation of a certain number of nearest sample, where the concept of near is realted to the eucildean distance between rows. 

**impute_all.py** This file generates a csv file that containes the imputed data. It obviously relies on the above-defined method, but it also performs some data preprocessing, that is important since every missing values, both Nan and numerical, has to be put to -1 before that the matrix is sent to the data_imputation fucntion.

**run.py** This file allows the reproduction of the results obtained in AICrowd submission