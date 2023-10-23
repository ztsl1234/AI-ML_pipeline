
# Machine Learning Pipeline

## Problem Statement

## Objectives

Your objective is to predict the occurrence of car failure using the provided dataset for an automotive company to formulate mitigative policies.

## (A) Full name (as in NRIC) and email address (stated in your application form).

### Name : Tan Siew Ling
### Email : ztsl1234@gmail.com

## (B) Overview of the submitted folder and the folder structure.

```bash
submitted folder
|____src
     |____config
     |____data
     |____models
     |____utils

```
### Folders:

(1) submitted folder - contains:

- eda.ipynb - jupyter notebook for Task 1 - Exploratory Data Analysis (EDA)

- requirements.txt

- run.sh - script to run the pipeline

- .gitignore - to define which file to exclude from the repository (e.g. data/failure.db)

(2) src - contains :
- MLPipeline.py - main program to run the pipeline

(3) src/config - contains :

- config file config.yml

(4) src/data - contains:

- DataManager.py : class to do all data related tasks e.g. retreiving data
 
- DataPrep.py : class to do all data cleaning and preparation tasks

(5) src/models - contains:

- ModelManager.py : class to do all modelling related tasks
 
- models.py : contains all the default models and parameters

(6) src/utils - contains:

- utils.py : contains all utility functions

- constants.py : contains all the constants 

## (C) Instructions for executing the pipeline and modifying any parameters.


### Step 1 : Train and Predict using the available models with default parameters

#### (a) Specified in the config.yml file.

#### config.yml

```bash
---
data_source: data/failure.db
models_to_use:
  - RandomForestClassifier
  - DecisionTreeClassifier
  - KNeighborsClassifier
  - GaussianNB
  - GradientBoostingClassifier
  - Perceptron
 
``` 

#### (b) Run the following command from the submitted folder  :

```bash
.\run.sh -all
```

#### Output:

- A csv file **all_scores.csv** will be generated containing the score results of various models.

- A log file **MLPipeline-all.log** will also be generated.

#### Results:
 
![image](https://user-images.githubusercontent.com/45007601/212603254-6d87929f-b94d-4fdd-89fe-18f20e1e692d.png)

Using Accuracy as a metric of evaulation :
We can see that the Top 4 models with highest Test Accuracy is :
  - RandomForestClassifier
  - KNeighborsClassifier
  - GaussianNB
  - GradientBoostingClassifier

![image](https://user-images.githubusercontent.com/45007601/212604056-f99fa34a-ddb2-4a70-b7cf-ac18e306c2f5.png)

There is some over-fitting for these models:
  - RandomForestClassifier
  - DecisionTreeClassifier

### Step 2 : Do Cross Validation for the Top 4 Models identified in Step 1 to determine if these models are robust enough

#### (a) Specify these 4 models in config.yml:

#### config.yml
```bash
---
data_source: data/failure.db
models_to_use:
  - RandomForestClassifier
  - KNeighborsClassifier
  - GaussianNB
  - GradientBoostingClassifier
 
``` 
#### (b) Run the following command from the submitted folder  :

```bash
.\run.sh -cv
```

#### Output:
- A csv file **cv_results.csv** will be generated containing the cross validation results of the models.

- A log file **MLPipeline-cv.log** will also be generated.

#### Results:

- The models with the least Cross Validation Errors indicates these models are most robust

![image](https://user-images.githubusercontent.com/45007601/212604407-c5f2dfd6-7747-4675-9c6c-a911c5558c37.png)

![image](https://user-images.githubusercontent.com/45007601/212604504-33570edc-8599-4bd7-92e9-c6c3022c60b6.png)

The mean accuracy for these models remain the same during cross validation and the cross validation errors are very low. 
This indicates that the models are robust enough.

### Step 3: Do hyperparameters tuning to find the best parameters for these 4 models
As it wil take a long time to do hyperparameters tuning for all the selected models, specify one model at one time in the config.yml file

#### (a) Specify the model in config.yml:

#### config.yml
```bash
---
data_source: data/failure.db
models_to_use:
  - RandomForestClassifier
 ```
#### (b) Run the following command from the submitted folder  :

```bash
.\run.sh -tuning
```
#### Output:
- A csv file **tuning_results_YYYYMMDD_HHMMSS.csv** will be generated containing the cross validation results of the models.

- A log file **MLPipeline-tuning.log** will also be generated.
 
#### Results:

#### Best Parameters:

![image](https://user-images.githubusercontent.com/45007601/212645978-64ecdae9-2620-420a-8ea3-a650b13b7b99.png)

The model Gradient Boosting Classifier with the above best parameters gives the Highest score after tuning


### Help : To get help on the parameters to pass :

#### (a) Run the following command from the submitted folder  :
```bash
.\run.sh --help
```
#### Output :
- A log file **MLPipeline--help.log** will also be generated with the following contents

```bash
usage: MLPipeline.py [-h] [-all] [-cv] [-tuning]

optional arguments:
  -h, --help  show this help message and exit
  -all        to run all models
  -cv         to run Cross Validation
  -tuning     to run Hyperparameters Tuning
```

## (D) Description of logical steps/flow of the pipeline. If you find it useful, please feel free to include suitable visualization aids (eg, flow charts) within the README.

(1) Load Data 

(2) Data Preparation

- Data Cleaning

- Handle missing data
 
- Feature Engineering
 
- Drop Features
 
- Scale the data to a common scale
 
- Transform skewed data using log
 
- Dummy Encode the categorical data

(2) if -all parameter is passed:

- get list of models to use from config.yml

- for each model, get the model object r

- Use this model to train and predict the test set

- Score the predictions

- store scores into csv file

(3) if -cv parameters is passed:

- get list of models to use from config.yml

- for each model, get the model object 

- Do cross validation using this model object

- store scores into csv file

(4) if -tuning parameters is passed:

- get list of models to use from config.yml

- for each model, get the model object 

- Do hyperparameters tuning to find the best parameters for the model

- Use the model with best prameters to predict test set

- store scores into csv file

## (E) Overview of key findings from the EDA conducted in Task 1 and the choices made in the pipeline based on these findings, particularly any feature engineering. 

### Summary:

(1) 5 Target variables : need to combine into 1 Target variable - Failure (1 =Fail, 0=No Failure)

(2) Missing data - Membership is a Categorical features. Hence, impute with Mode.

(3) Create new features (Synthetic Features)
- Year and Model Number from Model
- Age : Assuming Year from Model is the year of Manufacture, Age is calculated from Current Year - Year of Manufacture
- Temperature value and Temperature units from Temperature (Mixture of Temperature in Celsius and Farenheit - Have to convert all values to Celsius)
- Factory City and Country from Factory 

(2) Drop features
- Car ID : for identification of car only, useless as a feature
- Temeprature Units : all has same value - Celsius symbol, useless as a feature
- RPM - almost no correlation to Target (Failure) (-0.01< correlation <0.01)
- Temperature_value -  almost no correlation to Target (Failure) (-0.01< correlation <0.01)
- Color : almost no correlation to Target (Failure) (-0.01< correlation <0.01)

(4) Handle Different range of data by using feature Scaling (StandardScalar) to scale all the values to the same common scale so that the results will not be distorted during modelling

(5) Skewed data - apply np.log to numeric feature 

(6) Unbalanced dataset
- 87.4% of the cars did not fail. Only 12.6% failed. There are more cars that did not fail than cars that fail. The dataset is unbalanced. This is not an accurate representation of the population and the train and test set may be a biased one. This will result in poor model performance. Do stratified sampling when splitting the dataset into Train and Test set to ensure that each set still retain the same ratio of failed and success.


## (F) Described how the features in the dataset are processed (summarized in a table)
![image](https://user-images.githubusercontent.com/45007601/212602780-9d244d9a-9a8b-45e2-a9d4-01253eba6c8c.png)
![image](https://user-images.githubusercontent.com/45007601/212587563-48611eee-50f7-4243-9b40-2f998988633a.png)


## (G) Explanation of your choice of models for each machine learning task.

![image](https://user-images.githubusercontent.com/45007601/212587232-376d7e44-dd9d-4141-876f-7cd19926bd8f.png)
![image](https://user-images.githubusercontent.com/45007601/212587318-5559883e-e090-4bd6-9457-ffc23c95f1ae.png)


## (H) Evaluation of the models developed. Any metrics used in the evaluation should also be explained.

(1) Classification Report and Confusion Matrix

Besides the Accuracy metrics, we should also look at Precision and Recall metrics as Accuracy is not a clear indication of the model's performance especially for inbalanced dataset like this dataset

**(a) Precision**	- the ratio of true positives to the sum of true and false positives (quantity of the right predictions)

**(b) Recall/Sensitivity/True Positive Rate** - the ratio of true positives to the sum of true positives and false negatives.(quantity of right predictions the model made concerning the total positive values present - false negative is low)

**F1 Score** - the weighted harmonic mean of precision and recall. The closer the value of the F1 score is to 1.0, the better the expected performance of the model is. It is good for imbalanced datasets.

**Support** - the number of actual occurrences of the class in the dataset. It doesn’t vary between models, it just diagnoses the performance evaluation process.

### Random Forest model:
```
==trained model : RandomForestClassifier
==Training set
***Classification report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      6166
           1       1.00      0.99      1.00       890

    accuracy                           1.00      7056
   macro avg       1.00      1.00      1.00      7056
weighted avg       1.00      1.00      1.00      7056

***Confusion Matrix:
[[6165    1]
 [   6  884]]
Accuracy (Train)=99.90%
==Test set
***Classification report:
              precision    recall  f1-score   support

           0       0.92      0.94      0.93      2643
           1       0.50      0.43      0.46       382

    accuracy                           0.87      3025
   macro avg       0.71      0.68      0.70      3025
weighted avg       0.87      0.87      0.87      3025

***Confusion Matrix:
[[2477  166]
 [ 217  165]]
Accuracy (Test)=87.34%
```
### For Random Forest model:
- there is some overfitting as the accuracy of training results are much better than the test results. This is likely due to the unbalanced dataset. 

- ### High Precision 

  - Precision is 92% indicating that true positives is high. This mean that the model predicts the positive case (failure) correctly 92% of the time.
  
- ### High Recall 

  - Recall is 94% indicating that false negatives are low. This mean that model rarely predicts the negative case (no failure) wrongly.

### Decision Tree Forest :
```
==trained model : DecisionTreeClassifier
==Training set
***Classification report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      6166
           1       1.00      0.99      1.00       890

    accuracy                           1.00      7056
   macro avg       1.00      1.00      1.00      7056
weighted avg       1.00      1.00      1.00      7056

***Confusion Matrix:
[[6166    0]
 [   7  883]]
Accuracy (Train)=99.90%
==Test set
***Classification report:
              precision    recall  f1-score   support

           0       0.93      0.90      0.91      2643
           1       0.43      0.50      0.46       382

    accuracy                           0.85      3025
   macro avg       0.68      0.70      0.69      3025
weighted avg       0.86      0.85      0.86      3025

***Confusion Matrix:
[[2386  257]
 [ 190  192]]
Accuracy (Test)=85.22%
```

### For Decision Tree Forest model, 
- there is some overfitting as the accuracy of training results are much better than the test results. This is likely due to the unbalanced dataset. 

- ### High Precision 

  - Precision is 93% indicating that true positives is high. This mean that the model predicts the positive case (failure) correctly 93% of the time.
  
- ### High Recall 

  - Recall is 90% indicating that false negatives are low. This mean that model predicts the negative case (no failure) correctly most of the time.

### K Nearest Neighbors model:
```
==trained model : KNeighborsClassifier
==Training set
***Classification report:
              precision    recall  f1-score   support

           0       0.91      0.99      0.95      6166
           1       0.84      0.34      0.48       890

    accuracy                           0.91      7056
   macro avg       0.88      0.66      0.72      7056
weighted avg       0.90      0.91      0.89      7056

***Confusion Matrix:
[[6110   56]
 [ 589  301]]
Accuracy (Train)=90.86%
==Test set
***Classification report:
              precision    recall  f1-score   support

           0       0.90      0.98      0.94      2643
           1       0.62      0.24      0.35       382

    accuracy                           0.89      3025
   macro avg       0.76      0.61      0.64      3025
weighted avg       0.86      0.89      0.86      3025

***Confusion Matrix:
[[2587   56]
 [ 289   93]]
Accuracy (Test)=88.60%
```

### For K Nearest Neighbors model,
 
- there is little overfitting as the accuracy of training results are about the same the test results. 

- ### High Precision 

  - Precision is 90% indicating that true positives is high. This mean that the model predicts the positive case (failure) correctly 98% of the time.
  
- ### High Recall 

  - Recall is 98% indicating that false negatives are few. This mean that model predicts the negative case (no failure) correctly most of the time

### Gaussian Navie Bayes model:

```
==trained model : GaussianNB
==Training set
***Classification report:
              precision    recall  f1-score   support

           0       0.89      1.00      0.94      6166
           1       1.00      0.14      0.24       890

    accuracy                           0.89      7056
   macro avg       0.94      0.57      0.59      7056
weighted avg       0.90      0.89      0.85      7056

***Confusion Matrix:
[[6166    0]
 [ 766  124]]
Accuracy (Train)=89.14%
==Test set
***Classification report:
              precision    recall  f1-score   support

           0       0.89      1.00      0.94      2643
           1       1.00      0.15      0.26       382

    accuracy                           0.89      3025
   macro avg       0.95      0.57      0.60      3025
weighted avg       0.90      0.89      0.86      3025

***Confusion Matrix:
[[2643    0]
 [ 326   56]]
Accuracy (Test)=89.22%
```
### For Gaussian Navie Baye model,
- there is no overfitting as the accuracy of training results are almost the same the test results. 

- ### High Precision 

  - Precision is 89% indicating that true positives is high. This mean that the model predicts the positive case (failure) correctly 90% of the time.
  
- ### High Recall 

  - Recall is 100% indicating that false negatives are 0!. This mean that model predicts the negative case (no failure) correctly all the time!

### Gradient Boosting model:
```
==trained model : GradientBoostingClassifier
==Training set
***Classification report:
              precision    recall  f1-score   support

           0       0.92      1.00      0.96      6166
           1       0.98      0.38      0.55       890

    accuracy                           0.92      7056
   macro avg       0.95      0.69      0.75      7056
weighted avg       0.93      0.92      0.91      7056

***Confusion Matrix:
[[6160    6]
 [ 552  338]]
Accuracy (Train)=92.09%
==Test set
***Classification report:
              precision    recall  f1-score   support

           0       0.92      1.00      0.96      2643
           1       0.97      0.40      0.57       382

    accuracy                           0.92      3025
   macro avg       0.95      0.70      0.76      3025
weighted avg       0.93      0.92      0.91      3025

***Confusion Matrix:
[[2639    4]
 [ 228  154]]
Accuracy (Test)=92.33%
```
### For Gradient Boosting model,
- there is overfitting as the accuracy of training results are much better than the test results. This is likely due to the unbalanced dataset. 

- ### High Precision 

  - Precision is 92% indicating that true positives is high. This mean that the model predicts the positive case (failure) correctly 92% of the time.
  
- ### High Recall 

  - Recall is 100% indicating that false negatives are 0. This mean that model predicts the negative case (no failure) correctly most of the time.
  -
### Perceptron model:
```
==trained model : Perceptron
==Training set
***Classification report:
              precision    recall  f1-score   support

           0       0.89      0.87      0.88      6166
           1       0.23      0.29      0.26       890

    accuracy                           0.79      7056
   macro avg       0.56      0.58      0.57      7056
weighted avg       0.81      0.79      0.80      7056

***Confusion Matrix:
[[5334  832]
 [ 635  255]]
Accuracy (Train)=79.21%
==Test set
***Classification report:
              precision    recall  f1-score   support

           0       0.89      0.86      0.88      2643
           1       0.24      0.30      0.27       382

    accuracy                           0.79      3025
   macro avg       0.57      0.58      0.57      3025
weighted avg       0.81      0.79      0.80      3025

***Confusion Matrix:
[[2281  362]
 [ 268  114]]
Accuracy (Test)=79.17%

```
### For Perceptron model,
- there is no overfitting as the accuracy of training results are almost the same the test results. 

- ### High Precision 

  - Precision is 89% indicating that true positives is high. This mean that the model predicts the positive case (failure) correctly 89% of the time.
  
- ### High Recall 

  - Recall is 86% indicating that false negatives are low. This mean that model predicts the negative case (no failure) correctly most of the time.

### Model Evaluation Summary:

![image](https://user-images.githubusercontent.com/45007601/212607151-38db7cb4-de06-4df2-abc0-2cc00d250104.png)

(1) A model with High Recall and High Precision is important for our classification problem. If the Recall is low, then false negatives will be high and more positive case (Failure) will not be predicted correctly. This is critical as failure to predict a car failure will result in car accidents, injuries and loss of lives.

![image](https://user-images.githubusercontent.com/45007601/212603254-6d87929f-b94d-4fdd-89fe-18f20e1e692d.png)

Using Accuracy as a metric of evaulation :
We can see that the Top 4 models with highest Test Accuracy is :
  - RandomForestClassifier
  - KNeighborsClassifier
  - GaussianNB
  - GradientBoostingClassifier

![image](https://user-images.githubusercontent.com/45007601/212604056-f99fa34a-ddb2-4a70-b7cf-ac18e306c2f5.png)

There is some over-fitting for these models:
  - RandomForestClassifier
  - DecisionTreeClassifier

#### (2) Cross Validation

![image](https://user-images.githubusercontent.com/45007601/212604407-c5f2dfd6-7747-4675-9c6c-a911c5558c37.png)

![image](https://user-images.githubusercontent.com/45007601/212604504-33570edc-8599-4bd7-92e9-c6c3022c60b6.png)

The mean accuracy for these models remain the same during cross validation and the cross validation errors are very low. 
This indicates that the 4 models are robust enough.



## (I) Other considerations for deploying the models developed.

Besides the performance of the model, there are other considerations when choosing the model to deploy

(1) Explainability of the results
- It is important that the results of the model are easy to understand and interpret

(2) Complexity
- A complex model can give better performance. However, it is harder to maintain and explain. This will increase the cost of building and mainaining the model in the long run.

(3) Size of the Dataset
- The amount of training data available should be considered when choosing a model. Some models requires a minimum size to achieve a good result while others works well with a small set of training data. For example, A KNN (K-Nearest Neighbors) model is works better with a small training set.

(4) Number of features 
- Most models can achieve better results with more features (which also increase the complexity of the model) while the performance of some models drop with high-dimensional dataset.

(5). Training time and cost
- How long it takes and how much it costs to train a model has an impact on the model to choose to deploy 
- a 98%-accurate model that costs $100,000 to train versusor a 97%-accurate model that costs $10,000.
- if it is critical/important to have near real-time updates, then frequent training is required and hence training cannot be long and expensive. (Example: recommendation system which requires to be updated constantly cannot afford long and expensive training)

(6) Prediction time
- To make decision in real-time (Example: self-driving system) would requires a model which can make prediction quickly (Example: KNN versus Decision Tree although Decision Tree takes longer to train)
