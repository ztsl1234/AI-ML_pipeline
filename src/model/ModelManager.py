
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

from model import models
from utils import utils

class ModelManager:
    """
        This class performs all task related to modelling - fit,predict, scoring, cross validation, Hyperparameters tuning

        ...

        Attributes
        ----------
        to be done

        Methods
        -------
        to be done
    """
    def __init__(self):

        #cols=["Model","Training Error","Training Score","Training Accuracy","Test Error","Test Score","Test Accuracy","Diff Accuracy","Precision","Recall","Specificity","F1 score"]
        cols=["Model","Training Error","Training Score","Training Accuracy","Test Error","Test Score","Test Accuracy","Diff Accuracy"]
        self.score_df=pd.DataFrame(columns=cols)

        self.models_to_use_dict=None
  
    def get_models(self):  
        #filter for model to use based on the model list defined in config file
        if self.models_to_use_dict is not None:
            return self.models_to_use_dict
            
        return_dict={}
        models_to_use=utils.get_config("config.yml","models_to_use")
        print(f'''models_to_use={models_to_use}''')

        models_dict=models.default_models_dict

        for model in models_to_use:
            model_obj=models_dict.get(model)
            if model_obj is not None:
                return_dict[model]= model_obj

        return return_dict    

    def cross_validation(self,X_train, y_train):
        print(f'''=========Cross Validation=============''')
        kfold = StratifiedKFold(n_splits=10) #config
       
        cv_results = []
        for name,model in self.get_models().items():
            cv_results.append(cross_val_score(model, X_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

        cv_means = []
        cv_std = []
        for cv_result in cv_results:
            cv_means.append(cv_result.mean())
            cv_std.append(cv_result.std())
        print(len(cv_means))
        print(len(cv_std))
        cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Model":self.get_models().keys()})
        cv_res.sort_values(by=["CrossValerrors"], ascending=True,inplace=True)
        print(cv_res)
        cv_res.to_csv("cv_results.csv")
        
        #chart
        g =sns.barplot(x='CrossValMeans', y='Model', data=cv_res, palette="pastel")
        #g = sns.barplot("CrossValMeans","Model",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
        g.set_xlabel("Mean Accuracy")
        g = g.set_title("Cross validation scores")
        plt.show()

    def hyperparameters(self,pipe,X_train, y_train,X_test, y_test):
        #hyperparameters tuning
        
        model_name=pipe.named_steps.get("model").__class__.__name__
        print(f'''=========Hyperparameters tuning for {model_name}=============''')
    
        params=models.params_dict.get(model_name)
        print(f'''params={params}''')
        grid = GridSearchCV(pipe, params, cv=2).fit(X_train, y_train) #config?
        print('Training set score: ' + str(grid.score(X_train, y_train)))
        print('Test set score: ' + str(grid.score(X_test, y_test)))

        train_predictions=grid.predict(X_train)
        test_predictions=grid.predict(X_test)

        print(f'''=========Best Parameters=============''')
        
        # Access the best set of parameters
        best_params = grid.best_params_
                
        # Stores the optimum model in best_pipe
        best_pipe = grid.best_estimator_
        grid_score=grid.score(X_test,y_test)
        best_pipe_score=best_pipe.score(X_test,y_test)
        best_score=grid.best_score_
        
        result_df = pd.DataFrame.from_dict(grid.cv_results_, orient='columns')
        #print(result_df)
        result_df.to_csv(f"tuning_{model_name}.csv")

        model_name2=model_name + str(best_params)
        #self.scoring(grid,model_name2,train_predictions, test_predictions,X_train, y_train, X_test,y_test)
        
        print("Best params:",best_params)
        print("grid.score:",grid_score)
        print("grid best estimator:",best_pipe)
        print("grid best estimator score:",best_pipe_score)
        print("Best score:",best_score)
        
        grid_score_dict={}
        grid_score_dict["Model"]=model_name
        grid_score_dict["Best params"]=str(best_params)
        grid_score_dict["grid score"]=grid_score
        grid_score_dict["grid best estimator"]=best_pipe
        grid_score_dict["grid best estimator score"]=best_pipe_score
        grid_score_dict["Best score"]=best_score

        #print(f'''grid_score_dict={grid_score_dict}''')
        
        grid_score_df=pd.DataFrame(grid_score_dict,index=[0])
        grid_score_df.to_csv(f"grid_score_df{model_name}.csv",index=False)
        
        return grid_score_df

        """
        sns.relplot(data=result_df,
        kind='line',
        x='model__n_neighbors',
        y='mean_test_score',
        hue='param_scaler',
        col='param_classifier__p')
        plt.show()

        sns.relplot(data=result_df,
                    kind='line',
                    x='param_classifier__n_neighbors',
                    y='mean_test_score',
                    hue='param_scaler',
                    col='param_classifier__leaf_size')
        plt.show()
        """
    def fit_predict_score(self,pipeline,X_train, y_train, X_test,y_test):
        print(f'''=========Train and Predict =============''')

        pipeline.fit(X_train,y_train)
        model_name=pipeline.named_steps.get("model").__class__.__name__
        print(f'''==trained model : {model_name}''')
        print(f'''==predicting with model''')
        train_predictions=pipeline.predict(X_train)
        test_predictions=pipeline.predict(X_test)
        self.scoring(pipeline,model_name,train_predictions, test_predictions,X_train, y_train, X_test,y_test)

    def scoring(self,pipeline,model_name,train_predictions, test_predictions,X_train, y_train, X_test,y_test):
        print(f'''========= Scoring =============''')
        
        expected=y_train
        predicted=train_predictions

        train_mean_err=mean_absolute_error(predicted,expected)
        train_pipe_score=pipeline.score(X_train,expected)
        train_accuracy=metrics.accuracy_score(expected, predicted)*100

        score_list=[]
        #score_list.append(pipeline.named_steps.get("model").__class__.__name__)
        score_list.append(model_name)
        score_list.append(train_mean_err)
        score_list.append(train_pipe_score)
        score_list.append(train_accuracy)

        expected=y_test
        predicted=test_predictions

        test_mean_err=mean_absolute_error(predicted,expected)
        test_pipe_score=pipeline.score(X_test,expected)
        test_accuracy=metrics.accuracy_score(expected, predicted)*100
        #test_precision = metrics.precision_score(expected, predicted)
        #test_Sensitivity_recall = metrics.recall_score(expected, predicted)
        #test_Specificity = metrics.recall_score(expected, predicted, pos_label=0)
        #test_F1_score = metrics.f1_score(expected, predicted)

        score_list.append(test_mean_err)
        score_list.append(test_pipe_score)
        score_list.append(test_accuracy)
        score_list.append(test_accuracy-train_accuracy)
        #score_list.append(test_precision)
        #score_list.append(test_Sensitivity_recall)
        #score_list.append(test_Specificity)
        #score_list.append(test_F1_score)

        print(f'''score_list={score_list}''')

        self.score_df.loc[len(self.score_df )] = score_list

        #print(f'''self.score_df={self.score_df}''')
  
        print(f'''Training error: {train_mean_err}''')
        print(f'''Test error    : {test_mean_err}''' )

        print(f'''Training set score: {train_pipe_score}''')
        print(f'''Test set score    : {test_pipe_score}''')

    
        # summarize the fit of the model
        print(f'''=========fit of model =============''')
        print(f'''==Training set''')
        expected=y_train
        predicted=train_predictions

        print("***Classification report:")
        print(metrics.classification_report(expected, predicted))
        print("***Confusion Matrix:")
        print(metrics.confusion_matrix(expected, predicted))
        print("Accuracy (Train)={:.2f}%".format(train_accuracy))

        #confusion_matrix = metrics.confusion_matrix(expected, predicted)
        #cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
        #cm_display.plot()
        #plt.show()

        print(f'''==Test set''')
        expected=y_test
        predicted=test_predictions
        
        print("***Classification report:")
        print(metrics.classification_report(expected, predicted))
        print("***Confusion Matrix:")
        print(metrics.confusion_matrix(expected, predicted))
        print("Accuracy (Test)={:.2f}%".format(test_accuracy))

        #confusion_matrix = metrics.confusion_matrix(expected, predicted)
        #cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
        #cm_display.plot()
        #plt.show()

        """
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        """