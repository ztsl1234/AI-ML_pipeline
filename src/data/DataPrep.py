import sqlite3
import pandas as pd
import numpy as np
import datetime

class DataPrep:
    """
        This class handle all Data Cleaning and Data Preparation tasks

        ...

        Attributes
        ----------
       - to be done

        Methods
        -------
        - to be done
    """
    def __init__(self, df):
        self.num_vars=None
        self.categorical_vars=None
        self.df=df

        self.get_features(df)

    def preprocess(self):  
        '''
        process the data and return the processed data

        '''     
        print(f''' preprocessing ...df={self.df.shape}''')

        self.df=(self.df.pipe(self.special_cleaning)
              .pipe(self.erroneous_data)
              .pipe(self.missing_data)
              .pipe(self.clip_outliers)
              .pipe(self.feature_engine)
              .pipe(self.drop_features)
              .pipe(self.handle_diff_range)
              .pipe(self.handle_skewed_distribution)
              .pipe(self.encoding)
        )

        return self.df

    def get_features(self,df):
        # Get list of numeric and categorical features
        self.num_vars=list(df.columns[df.dtypes != "object"])
        self.categorical_vars=list(df.columns[df.dtypes=="object"])


    def special_cleaning(self,df):  
        """
        Data Cleaning - Combine the 5 Target variables to 1 Target variable
        """
        print("--------Special Cleaning---------")
        df["Failure"]=0
        df.loc[df["Failure A"]==1,["Failure"]]=1
        df.loc[df["Failure B"]==1,["Failure"]]=1
        df.loc[df["Failure C"]==1,["Failure"]]=1
        df.loc[df["Failure D"]==1,["Failure"]]=1
        df.loc[df["Failure E"]==1,["Failure"]]=1

        #drop original target variables
        drop_cols=["Failure A","Failure B","Failure C","Failure D","Failure E"]
        df=df.drop(drop_cols,axis=1)

        print(df.info())

        return df
    def drop_features(self,df):  
        """
        Drop features with low correlation
        """
        print("--------drop features---------")
 
        cols=["Car ID","Color","RPM","Temperature_value"]
        df = df.drop(columns=cols)
        print(df.info())      
        return df

    def erroneous_data(self,df):  
        """
        lean up inconsistent data
        """
        print("--------clean erroneous data ---------")

        print("No erroneous data...")
  
        return df

    def missing_data(self,df):  
        """
        handle missing data
 
        (1) Missing data - Membership is a Categorical features. Hence, we can only impute with Mode.
        """        
        print("--------handle missing data ---------")
        # replace with mode
        feature="Membership"
        mode=df[feature].mode()[0]
        df[feature].fillna(mode, inplace=True)

        print(f"imputing {feature} with {mode}")

        return df

    def clip_outliers(self, df):
        """
        remove outliers
        """        
        print("--------Clipping outliers ---------")        

        print("not clipping outliers...")

        return df
    
    #z-score,min max => Scalers
    def handle_diff_range(self,df):
        """
        handle diff ranges in data
        """        
        print("--------Handle Different Range in data---------")  

        print("to be handled in pipeline...") 

        return df

    def handle_skewed_distribution(self,df):
        """
         log transformation
        """
         
        print("--------Handle skewed distribution---------")  
        ###accuracy drop if apply this
        
        print(df.columns)
        self.get_features(df)
        self.num_vars.remove("Failure")

        for feature in self.num_vars:
            df[feature]=np.log(df[feature])  
        
        return df
    
    def encoding(self,df):
        """
        Dummy encode all categorical features
        
        """         
        print("--------Encode Categorical features ---------")
        #Encode all the categorical features 
        #so that they can be processed by the learning algorithms
        self.get_features(df)
        df = pd.get_dummies(df,columns=self.categorical_vars)

        return df
    

    def feature_engine(self,df):
        """
         Create new features (Synthetic Features)
        """         
        print("--------Feature Engineering ---------")        
        df=(df.pipe(self.create_age)
              .pipe(self.create_factory_city_country)
              .pipe(self.create_temperature_value)
        )

        print(df.info()) 
        return df

    def create_age(self,df):
        """
         Create new features - Age, Year of Manufacture
        """         
        print("--------Create  Age, Year of Manufacture ---------")   

        current_year=datetime.datetime.now().year
        print(f'''current_year={current_year}''')

        df[["Model_num","Year_of_Manufacture"]]= df["Model"].str.split(",",expand=True)
        df["Year_of_Manufacture"]=df["Year_of_Manufacture"].astype(int)

        df["Age"]= current_year - df["Year_of_Manufacture"]

        df=df.drop("Model",axis=1)

        return df

    
    def create_temperature_value(self,df):
        """
        Create new feature - Temperature value from Temperature
        -  
        (1) Temperature feature is a combination of 2 features - temperature value and units.

        split the feature into 2 features - Temp_value (float) an Temp_units (string/object)
        (2) Mixture of values in Fahrenheit and Celsius

        convert Fahrenheit to Celsius using formula : Celsius =(Fahrenheit − 32) × 5/9 so that all temperature value will be in Celsius
        """         
        print("--------Create  Temperature value ---------")   
                 
        df[["Temperature_value","Temperature_units"]]=df["Temperature"].str.split(" ",expand=True)

        #convert
        df["Temperature_value"]=df["Temperature_value"].astype(float)
        df.loc[df["Temperature_units"]=="°F", ["Temperature_value"]]= (df["Temperature_value"] - 32) *(5/9)
        df.loc[df["Temperature_units"]=="°F", ["Temperature_units"]]= "°C"

        df=df.drop(columns=["Temperature","Temperature_units"],axis=1)

        return df
    
    def create_factory_city_country(self,df):
        """
        Create new features - Factory City and Country from Factory feature
        """         
        print("--------Create  Factory City and Country ---------")   
                
        df[["Factory_City","Factory_Country"]]= df["Factory"].str.split(",",expand=True)

        df=df.drop("Factory",axis=1)

        return df


