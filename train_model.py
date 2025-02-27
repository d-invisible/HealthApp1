import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from utils.preprocess import preprocess_data

# Load dataset
data = pd.read_csv("data/food_data.csv")

# Preprocess data
X, scaler, encoder = preprocess_data(data)

# Define suitability score based on nutrient thresholds
def calculate_suitability_score(row):
    """
    Calculate suitability score based on nutrient values.
    Example: High protein and low carbs are considered suitable.
    """
    protein = row["Protein (g)"]
    carbs = row["Carbohydrates (g)"]
    fat = row["Fat (g)"]
    
    # Example logic: High protein (>30g) and low carbs (<50g) are suitable
    if protein > 30 and carbs < 50:
        return 1  # Suitable
    else:
        return 0  # Not suitable

# Apply the function to create the target variable
data["Suitability_Score"] = data.apply(calculate_suitability_score, axis=1)
y = data["Suitability_Score"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
input_layer = Input(shape=(X.shape[1],))
x = Dense(128, activation="relu")(input_layer)
x = Dropout(0.2)(x)
x = Dense(64, activation="relu")(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Save model
model.save("models/meal_recommendation_model.h5")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(encoder, "models/encoder.pkl")