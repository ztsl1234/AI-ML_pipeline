import pandas as pd
import numpy as np
import yaml
import os


def get_config(yaml_file:str,name:str) :  
    '''
    Returns the value for a configuration

    Parameters:
            a (int): A decimal integer
            b (int): Another decimal integer

    Returns:
            binary_sum (str): Binary string of the sum of a and b
    '''     
    dir=os.getcwd()
    
    yaml_fullpath=f"{dir}/src/config/{yaml_file}"
    stream = open(yaml_fullpath, 'r')
    yaml_dict = yaml.full_load(stream)
    for key, value in yaml_dict.items():
        print (key + " : " + str(value))
        
    return yaml_dict.get(name)

 
     

