from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import os
import joblib
import yaml
from datetime import datetime
import util as util
import numpy as np
# import warnings for ignore the warnings
import warnings 
warnings.filterwarnings("ignore")
# import pickle and json file for columns and model file
import pickle
import json
import copy


def read_raw_data(config: dict) -> pd.DataFrame:
    # Create variable to store raw dataset
    raw_dataset = pd.DataFrame()

    # Raw dataset dir
    raw_dataset_dir = config["raw_dataset_dir"]

    # Look and load add CSV files
    for i in tqdm(os.listdir(raw_dataset_dir)):
        raw_dataset = pd.concat([pd.read_csv(raw_dataset_dir + i), raw_dataset])
    
    # Return raw dataset
    return raw_dataset


######################################################################################################################    

if __name__ == "__main__":
    # Load configuration file
    config_data = util.load_config()
    
    # Read all raw Dataset
    raw_dataset = read_raw_data(config_data)
    
    ## drop baris dengan label None
    raw_dataset.drop(raw_dataset[raw_dataset.category=='None'].index, inplace=True)
    
    # Splitting input output
    X = raw_dataset.drop(columns="category")
    y = raw_dataset["category"]

    # Splitting train test
    #Split Data 80% training 20% testing
    X_train, X_test, \
    y_train, y_test = train_test_split(
        X, y, 
        test_size = 0.2, 
        random_state = 42)
    
    # Splitting test valid
    X_valid, X_test, \
    y_valid, y_test = train_test_split(
        X_test, y_test,
        test_size = 0.4,
        random_state = 42,
        stratify = y_test
    )
    
    #Menggabungkan x train dan y train untuk keperluan EDA
    util.pickle_dump(X_train, config_data["train_set_path"][0])
    util.pickle_dump(y_train, config_data["train_set_path"][1])

    util.pickle_dump(X_valid, config_data["valid_set_path"][0])
    util.pickle_dump(y_valid, config_data["valid_set_path"][1])

    util.pickle_dump(X_test, config_data["test_set_path"][0])
    util.pickle_dump(y_test, config_data["test_set_path"][1])
    
    print("Data Pipeline passed successfully.")
