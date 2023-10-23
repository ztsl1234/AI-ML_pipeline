
import pandas as pd
import numpy as np

#models
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier # The k-nearest neighbor classifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron

import matplotlib.pyplot as plt # For plotting data
import seaborn as sns # For plotting data

from sklearn.model_selection import train_test_split # For train/test splits
from sklearn.feature_selection import VarianceThreshold # Feature selector
from sklearn.pipeline import Pipeline # For setting up pipeline


from sklearn.model_selection import train_test_split

# Various pre-processing steps
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, PowerTransformer, MaxAbsScaler, LabelEncoder, OneHotEncoder


# Cross validate model with Kfold stratified cross validation
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV # For optimization

from sklearn.metrics import mean_absolute_error
from sklearn import metrics

random_state=42

logreg = LogisticRegression(LogisticRegression(penalty='elasticnet'))
randForest=RandomForestClassifier(random_state = random_state)
decTree=DecisionTreeClassifier(random_state = random_state)
knn=KNeighborsClassifier()
svc=SVC(random_state=random_state)
gaussianNB=GaussianNB()
gradBoosting=GradientBoostingClassifier()        
percept=Perceptron(random_state=random_state)
linSVC=LinearSVC(random_state=random_state)
sgd=SGDClassifier(random_state=random_state)

default_models_dict={}
default_models_dict["LogisticRegression"]=logreg
default_models_dict["RandomForestClassifier"]=randForest
default_models_dict["DecisionTreeClassifier"]=decTree
default_models_dict["KNeighborsClassifier"]=knn
default_models_dict["SVC"]=svc
default_models_dict["GaussianNB"]=gaussianNB
default_models_dict["GradientBoostingClassifier"]=gradBoosting
default_models_dict["Perceptron"]=percept
default_models_dict["LinearSVC"]=linSVC
default_models_dict["SGDClassifier"]=sgd


params_KNN = {
'scaler': [StandardScaler(), MinMaxScaler(), Normalizer(), MaxAbsScaler()],
#'selector__threshold': [0, 0.001, 0.01],
'model__n_neighbors': [1, 3, 5, 7, 10],
'model__p': [1, 2],
'model__leaf_size': [1, 5, 10, 15]
}

params_random_forest = {
'scaler': [StandardScaler(), MinMaxScaler(), Normalizer(), MaxAbsScaler()],
#'selector__threshold': [0, 0.001, 0.01],
'model__n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
#'model__max_features':  ['auto', 'sqrt'],
#'model__max_depth': [int(x) for x in np.linspace(10, 110, num = 11)] + ["None"],
#'model__min_samples_split':[2, 5, 10],
#'model__min_samples_leaf': [1, 2, 4],
##'model__bootstrap': [True, False]
}

params_GNB = {
'scaler': [StandardScaler(), MinMaxScaler(), Normalizer(), MaxAbsScaler()],
#'selector__threshold': [0, 0.001, 0.01],
'model__var_smoothing': np.logspace(0,-9, num=100),
}

params_GBC={
'scaler': [StandardScaler(), MinMaxScaler(), Normalizer(), MaxAbsScaler()],
#'selector__threshold': [0, 0.001, 0.01],
'model__n_estimators': [ 1, 2, 4, 8, 16, 32, 64, 100, 200],
#'model__max_depths': np.linspace(1, 32, 32, endpoint=True),
#'model__min_samples_splits': np.linspace(0.1, 1.0, 10, endpoint=True),
#'model__min_samples_leafs':  np.linspace(0.1, 0.5, 5, endpoint=True)
}

params_dict={}
#params_dict["LogisticRegression"]=params_logreg
params_dict["RandomForestClassifier"]=params_random_forest
#params_dict["DecisionTreeClassifier"]=params_dec_tree
params_dict["KNeighborsClassifier"]=params_KNN
#params_dict["SVC"]=svc
params_dict["GaussianNB"]=params_GNB
params_dict["GradientBoostingClassifier"]=params_GBC
#params_dict["Perceptron"]=params_Percep
#params_dict["LinearSVC"]=params_linSVC
#params_dict["SGDClassifier"]=params_sgd

####best params model
best_params_default_models_dict={}
