import pandas as pd
import numpy as np
import argparse
import sys
import datetime

from model.ModelManager import ModelManager
from data.DataManager import DataManager
from data.DataPrep import DataPrep

import matplotlib.pyplot as plt # For plotting data
import seaborn as sns # For plotting data

from sklearn.model_selection import train_test_split # For train/test splits
from sklearn.feature_selection import VarianceThreshold # Feature selector
from sklearn.pipeline import Pipeline # For setting up pipeline


from sklearn.model_selection import train_test_split

# Various pre-processing steps
from sklearn.preprocessing import FunctionTransformer, Normalizer, StandardScaler, MinMaxScaler, PowerTransformer, MaxAbsScaler, LabelEncoder, OneHotEncoder
#from sklearn.impute import SimpleImputer

class MLPipeline:
    def __init__(self,all,cv,tuning):
        print(f'''=========Initializing =============''')

        if not (all or cv or tuning):
            print(f''' Invalid option. Please provide option -all, -cv, -tuning or --help. Nothing to do. Quiting... ''')      
            sys.exit(0)

        #Load Data
        self.df=DataManager().load_data()
        print(self.df.head())

        self.model_mgr=ModelManager()
        self.model_dict=self.model_mgr.get_models()

        self.cv=cv
        self.tuning=tuning
        self.all=all

    def run(self) -> None:
        print(f'''=========Start=============''')
     
        self.df=DataPrep(self.df).preprocess()

        y = self.df["Failure"]
        X = self.df.drop(["Failure"],axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42,stratify=y)

        if self.cv:
            self.model_mgr.cross_validation(X_train, y_train)

        elif self.tuning:
            score_df=pd.DataFrame([])           
            for name,model in self.model_mgr.get_models().items():

                print(f'''=========Tuning model : {name}=============''')
                print(model)

                steps = [
                        #('Encoder',OneHotEncoder()),
                        ('scaler', StandardScaler()), #normal distr MinMaxScaler(), Normalizer() and MaxAbsScaler().
                        ('selector', VarianceThreshold()),
                        ("model", model)]

                pipe = Pipeline(steps)

                s_df=self.model_mgr.hyperparameters(pipe,X_train, y_train,X_test, y_test)
                score_df=pd.concat([score_df,s_df])

            #print score report
            dt=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            #score_df.sort_values(by="Test Accuracy",ascending=False, inplace=True)
            print(score_df)
            score_df.to_csv(f"tuning_results_{dt}.csv")
    
        elif self.all:
            for name,model in self.model_mgr.get_models().items():

                print(f'''=========running with model : {name}=============''')
                print(model)

                #transformer = FunctionTransformer(np.log1p)

                steps = [
                        #('Encoder',OneHotEncoder()),
                        ('scaler', StandardScaler()), #normal distr MinMaxScaler(), Normalizer() and MaxAbsScaler().
                        #('transformer', transformer), #skewed data.
                        ('selector', VarianceThreshold()),
                        ("model", model)]

                pipe = Pipeline(steps)

                self.model_mgr.fit_predict_score(pipe,X_train, y_train, X_test,y_test)
            
            #print score report
            self.model_mgr.score_df.sort_values(by="Test Accuracy",ascending=False, inplace=True)
            #print(self.model_mgr.score_df[["Model","Training Accuracy","Test Accuracy","Diff Accuracy","Precision","Recall","Specificity","F1 score"]])
            print(self.model_mgr.score_df[["Model","Training Accuracy","Test Accuracy","Diff Accuracy"]])
            self.model_mgr.score_df.to_csv("all_scores.csv")

            #chart
            g =sns.barplot(x='Test Accuracy', y='Model', data=self.model_mgr.score_df, palette="pastel")
            g.set_xlabel("Test Accuracy")
            g = g.set_title("Accuracy")
            plt.show()
  
        else:
            print(f'''option not valid. Nothing to do. Quiting... ''')        


    def log_transform(self,x):
        print(x)
        return np.log(x + 1)



#main

parser=argparse.ArgumentParser()

parser.add_argument('-all',  dest='all', action='store_true', default=False,help='to run all models')
parser.add_argument('-cv',  dest='cv', action='store_true', default=False,help='to run Cross Validation')
parser.add_argument('-tuning', dest='tuning', action='store_true', default=False,help='to run Hyperparameters Tuning') 


args = parser.parse_args()

crossv=args.cv
hyperp=args.tuning
all=args.all

mlpipe=MLPipeline(all,crossv,hyperp)
mlpipe.run()
