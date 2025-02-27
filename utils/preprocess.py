import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(data):
    # Numerical features
    numerical_features = data[["Calories (kcal)", "Protein (g)", "Carbohydrates (g)", "Fat (g)", "Fiber (g)", "Sugars (g)", "Sodium (mg)", "Cholesterol (mg)"]]
    
    # Categorical features
    categorical_features = data[["Category", "Meal_Type"]]
    
    # Normalize numerical features
    scaler = StandardScaler()
    numerical_features = scaler.fit_transform(numerical_features)
    
    # Encode categorical features
    encoder = OneHotEncoder(sparse=False, drop="first")
    categorical_features = encoder.fit_transform(categorical_features)
    
    # Combine features
    X = np.concatenate((numerical_features, categorical_features), axis=1)
    
    return X, scaler, encoder