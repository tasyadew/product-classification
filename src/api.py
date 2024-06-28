from fastapi import FastAPI, Form
from pydantic import BaseModel
import pandas as pd
from joblib import load
import joblib
import function_1_data_pipeline as function_1_data_pipeline
import function_2_data_processing as function_2_data_processing
import function_3_modeling as function_3_modeling
# import src.function_1_data_pipeline as function_1_data_pipeline
# import src.function_2_data_processing as function_2_data_processing
# import src.function_3_modeling as function_3_modeling
import util as util
import requests
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle
from tqdm import tqdm
import os
import copy
import yaml
from datetime import datetime
import uvicorn
import sys

#API
app = FastAPI() 
config_data = util.load_config()
tfidf_vectorizer = function_2_data_processing.load_tfidf_vectorizer(config_data["tfidf_vectorizer"])
model = joblib.load(config_data["model_final"])

class api_data(BaseModel):
    title: str

@app.get("/")
def home():
    return "Hello, FastAPI up!"    

@app.post("/predict/")
def predict(data: api_data):
    # Convert data api to dataframe
    config_data = util.load_config()
    
    #Input data
    df = pd.DataFrame(data.dict(), index=[0])
    df = function_2_data_processing.preprocess_text(df, "title")
    df = tfidf_vectorizer.transform(df['title'])

    predicted_class = model.predict(df)
    predicted_class = predicted_class[0]

    class_mapping = {
        0: "Air fresheners",
        1: "Appliance",
        2: "Baby needs",
        3: "Bath & Lotion",
        4: "Bath & Towel",
        5: "Bedding",
        6: "Biscuit & Cookies",
        7: "Body care",
        8: "Bread",
        9: "Butter & Creams",
        10: "Car care & Accessories",
        11: "Carbonated & Packed drink",
        12: "Cereal & Breakfast",
        13: "Cheese",
        14: "Chips & Crisps",
        15: "Chocolate & Candy",
        16: "Chocolate & Nutritious drink",
        17: "Cleaning & Tools",
        18: "Coffee",
        19: "Cooking & Baking ingredients",
        20: "Cooking oil",
        21: "Cosmetic",
        22: "Cutlery",
        23: "Deodorants",
        24: "Desserts & Ice cream",
        25: "Diapers & Wipes",
        26: "Dinnerware",
        27: "Dish washing",
        28: "Drinkware",
        29: "Dry & Canned food",
        30: "Eggs",
        31: "Fish & Seafood",
        32: "Food",
        33: "Food storage",
        34: "Frozen food",
        35: "Fruits",
        36: "Gardening",
        37: "Hair care",
        38: "Hand care",
        39: "Hardware",
        40: "Health & Safety",
        41: "Health & Wellness",
        42: "Home interior",
        43: "Juices",
        44: "Juices & Cordial",
        45: "Kitchen organiser",
        46: "Kitchen utensil",
        47: "Laundry",
        48: "Meat & Poultry",
        49: "Milk",
        50: "Milk & Creamers",
        51: "Milk powder",
        52: "Non halal",
        53: "Noodles",
        54: "Noodles & Pasta",
        55: "Nuts & Seeds",
        56: "Oral care",
        57: "Pest control",
        58: "Pet food",
        59: "Pot & Pan",
        60: "Preserve foods",
        61: "Rice and grains",
        62: "Sanitary",
        63: "Sauce & Paste",
        64: "Shaving & Grooming",
        65: "Skin care",
        66: "Spices & Dry condiments",
        67: "Sports & Outdoor",
        68: "Stationery",
        69: "Sugars & Sweeteners",
        70: "Tea",
        71: "Tissue",
        72: "Tofu",
        73: "Toys",
        74: "Vegetable",
        75: "Water",
        76: "Yogurt & Pudding"
    }

    if predicted_class in class_mapping:
        return class_mapping[predicted_class]
    else:
        return "Unknown Category"

    
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8500, reload=True)
