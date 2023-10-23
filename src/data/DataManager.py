import sqlite3
import pandas as pd
import numpy as np
import os

from utils  import utils
from utils import constants

class DataManager:
    """
        A class to manage all data related tasks e.g. connection, retrieval

        ...

        Attributes
        ----------
        None

        Methods
        -------
        - to be done
    """
    def __init__(self):
        # Create your connection.
        #data_file="data/failure.db" #in config
  
        data_file=utils.get_config("config.yml","data_source")

        #data_file=f"/data/failure.db"
        print(f"Connecting to {data_file}")
        self.sqliteConnection = sqlite3.connect(data_file)
        print(f"Connected to SQLite - {data_file}")

    def load_data(self) -> pd.DataFrame:  
        '''
        Returns the data that is loaded.

        Parameters:
            None

        Returns:
            dataframe of the data
        '''     
        df = pd.read_sql_query("SELECT * FROM failure", self.sqliteConnection)  

        return df

 
     

