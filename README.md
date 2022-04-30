# Neural Network to Predict Success of Funds Recipient

## Overview

Not all funds donated to charitable organizations are impactful. While some bring the intended changes, others may be simply money wasted. The objective of this anaysis is to use Neural Network and Deep Learning to predict success of funds recipients based on a number of characteristics including application type, affiliation, use case, organization type, income, etc.


### Resources

- Data Source: charity_data.csv
- Software: Python 3.7.6, Jupyter Notebook

### Purpose

The main purposes are:

- preprocessing the data for the neural network model,
- compile, train and evaluate the model,
- optimize the model.


## Results

### Data Preprocessing

- The column **IS_SUCCESSFUL** contains binary data refering whether the charity donation was used effectively. Therefore, this variable is considered as the target for the neural network model.

- The following columns **APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT** are the features for the model.

- The columns **EIN and NAME** are not meaningful for the analysis (not a feature) and hence removed from the input data.

- For those columns that have more than 10 unique categories, number of data points per category was calculated and the data inspected using density plots. Categories with very few records were grouped together as an _other_ category.

- All categorical variables were encoded for the analysis, data was split into training and testing sets, and features were standardized.


### Compiling, Training, and Evaluating the Model

- The input data has **43** features and **25,724** samples.

- For this deep learning neural network model, two hidden layers with **80** and **30** neurons were selected respectively.

- **ReLU** was used as the activation function for both hidden layers. As the output is binary, **Sigmoid** was used for the output layer.

- For the compilation, the optimizer chosen was **adam** and the loss function was **binary_crossentropy**.

- The **Accuracy** for training and testing dataset are approximately **74%** and **73%** respectively, which were not able to achieve the target model performance of **75%**. This falls short of satisfyingly predict the outcome of the charity donations.

- The result was saved and exported to an HDF5 file as **AlphabetSoupCharity.h5**.

![Accuracy]()


### Optimizing the Model

To increase the performance of the model, a few measures were taken. They are described below.

#### 1. Removing of unimportant variables

- In the first attempt, **STATUS** and **SPECIAL_CONSIDERATIONS** columns were removed. By using cross-tabulation, the proportion of occurance of successful and unsucessful fund recipients were found to be more or less same accross different categories of these two features. This implies that the two features don't add value to the prediction much but can contribute to overfitting.

- To increase the performance of the model, we applied bucketing to the feature **ASK_AMT** by labeling ask amount as **low** for anything equal to or less than 10,000 and **high** otherwise. This was done since there were very large values beyond Q3 of this variable which could have a very different relationship with the target variable.

- **ReLU** was used as the activation function for both hidden layers. **Sigmoid** was used for the output layer.

![Accuracy1]()


- However, this did not really improve the accuracy which stayed at **74%** for training and **73%** for testing dataset.


#### 2. Increasing hidden layers and number of neurons

- In this attempt, one more hidden layer with additional neurons were added to see if the model performance could be improved. The three hidden layers had **90**, **50** and **30** neurons, respectively. The activation functions was **ReLU** for all the hidden layers **Sigmoid** was used for the output layer. The model was then trained and tested as before.

![Accuracy2]()


- Unfortunately, the accuracy for training and testing dataset stayed at approximately **74%** and **72%**, respectively.


#### 3. Change activation function to hidden layers and increase number of neurons

- In this attempt, two hidden layes with **120** and **60** neurons were used in the model. 

- The **tanh** activation function used in all the hidden layers with **Sigmoid** for the output layer.

![Accuracy3]()

- This model returned a **74%** accuracy for training data but only **72%** for test data


#### 4. Random Forest

- A Random Forest model with a depth of 6 produces the best performance but no better than **73%** prediction accuracy.

![RandomForestAccuracy]


#### 5. Logistic Regression

- A Logistic Regression with 50 iterations produced a predictive accuracy of **72%**. 

![LogisticRegressionAccuracy]()


## Summary

A number of different models and variations in model specifications were tried to achieve the predertimed target prediction accuracy of 75%. However, it could not be met. Maybe inclusion of more relevant features could improve prediction accuracy further.