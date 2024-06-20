from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import copy
import joblib
import yaml
from datetime import datetime
import pandas as pd
import numpy as np
import json
import nltk
nltk.download('stopwords')
import warnings
warnings.filterwarnings("ignore")
from nltk.corpus import stopwords
import nltk
import re
import pickle
import datetime
import util as util
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the stopwords from the NLTK library
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
config_dir = "config/config.yaml"

############################################
def load_data(config_data: dict) -> pd.DataFrame:
    # Load every set of data
    X_train = util.pickle_load(config_data["train_set_path"][0])
    y_train = util.pickle_load(config_data["train_set_path"][1])

    X_valid = util.pickle_load(config_data["valid_set_path"][0])
    y_valid = util.pickle_load(config_data["valid_set_path"][1])

    X_test = util.pickle_load(config_data["test_set_path"][0])
    y_test = util.pickle_load(config_data["test_set_path"][1])

    # Return 3 set of data
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def load_dataset(config_data: dict) -> pd.DataFrame:
    # Load every set of data
    X_train = util.pickle_load(config_data["train_set_path"][0])
    y_train = util.pickle_load(config_data["train_set_path"][1])

    X_valid = util.pickle_load(config_data["valid_set_path"][0])
    y_valid = util.pickle_load(config_data["valid_set_path"][1])

    X_test = util.pickle_load(config_data["test_set_path"][0])
    y_test = util.pickle_load(config_data["test_set_path"][1])

    # Concatenate x and y each set
    train_set = pd.concat(
        [X_train, y_train],
        axis = 1
    )
    valid_set = pd.concat(
        [X_valid, y_valid],
        axis = 1
    )
    test_set = pd.concat(
        [X_test, y_test],
        axis = 1
    )

    # Return 3 set of data
    return train_set, valid_set, test_set


def preprocess_text(df, column_name):
    # Remove special characters and convert to lowercase
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x).lower())
    
    # Remove stopwords and join the words with a single space
    df[column_name] = df[column_name].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    
    return df


# Inisialisasi dan fit TfidfVectorizer
def fit_tfidf_vectorizer(X_train, save_path):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(X_train['title'])
    
    # Simpan TfidfVectorizer
    with open(save_path, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    
    return tfidf_vectorizer


# Load TfidfVectorizer dari file
def load_tfidf_vectorizer(file_path):
    with open(file_path, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    return tfidf_vectorizer


# Inisialisasi dan latih LabelEncoder
def fit_label_encoder(y_train, save_path):
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    
    # Simpan LabelEncoder ke dalam file
    with open(save_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    return label_encoder

# Load LabelEncoder dari file
def load_label_encoder(file_path):
    with open(file_path, 'rb') as f:
        label_encoder = pickle.load(f)
    return label_encoder


if __name__ == "__main__":
    # Load configuration file
    config_data = util.load_config()

    # Load dataset
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data(config_data)
    
    # Membersihkan teks dari karakter khusus dan mengonversi teks menjadi huruf kecil
    X_train = preprocess_text(X_train, "title")
    X_test = preprocess_text(X_test, "title")
    X_valid = preprocess_text(X_valid, "title")
    
    # Inisialisasi dan latih TfidfVectorizer
    tfidf_vectorizer = fit_tfidf_vectorizer(X_train, config_data["tfidf_vectorizer"])
    
    # Transformasi data pelatihan
    X_train_tfidf = tfidf_vectorizer.transform(X_train['title'])
    # Transformasi data uji
    X_test_tfidf = tfidf_vectorizer.transform(X_test['title'])
    # Transformasi data validasi
    X_valid_tfidf = tfidf_vectorizer.transform(X_valid['title'])
    
    # Inisialisasi dan latih LabelEncoder
    label_encoder = fit_label_encoder(y_train, config_data["label_encoder"])
    # Memuat LabelEncoder yang sudah terlatih dari file
    loaded_label_encoder = load_label_encoder(config_data["label_encoder"])

    # Anda dapat menggunakannya untuk melakukan transformasi pada data
    y_train_encoded = loaded_label_encoder.transform(y_train)
    y_test_encoded = loaded_label_encoder.transform(y_test)
    y_valid_encoded = loaded_label_encoder.transform(y_valid)

    # Save Data
    util.pickle_dump(X_train_tfidf, config_data["train_tfidf_set_path"][0])
    util.pickle_dump(y_train_encoded, config_data["train_tfidf_set_path"][1])

    util.pickle_dump(X_valid_tfidf, config_data["valid_tfidf_set_path"][0])
    util.pickle_dump(y_valid_encoded, config_data["valid_tfidf_set_path"][1])

    util.pickle_dump(X_test_tfidf, config_data["test_tfidf_set_path"][0])
    util.pickle_dump(y_test_encoded, config_data["test_tfidf_set_path"][1])