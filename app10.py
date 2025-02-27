import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the trained model
model = load_model("models/meal_recommendation_model.h5")

# Load the fitted scaler and encoder
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/encoder.pkl")

# Load dataset
data = pd.read_csv("data/food_data.csv")

# Load or create user feedback file
if not os.path.exists("feedback/user_feedback.csv"):
    feedback_data = pd.DataFrame(columns=["User_ID", "Food_Item", "Meal_Type", "Feedback"])
    feedback_data.to_csv("feedback/user_feedback.csv", index=False)
else:
    feedback_data = pd.read_csv("feedback/user_feedback.csv")

# Function to preprocess data
def preprocess_data(data, scaler=None, encoder=None, fit=False):
    """
    Preprocess the data for prediction or training.
    """
    # Numerical features
    numerical_features = data[["Calories (kcal)", "Protein (g)", "Carbohydrates (g)", "Fat (g)", "Fiber (g)", "Sugars (g)", "Sodium (mg)", "Cholesterol (mg)"]]
    
    # Categorical features
    categorical_features = data[["Category", "Meal_Type"]]
    
    # Normalize numerical features
    if scaler is None:
        scaler = StandardScaler()
    if fit:
        numerical_features = scaler.fit_transform(numerical_features)
    else:
        numerical_features = scaler.transform(numerical_features)
    
    # Encode categorical features
    if encoder is None:
        encoder = OneHotEncoder(sparse=False, drop="first")
    if fit:
        categorical_features = encoder.fit_transform(categorical_features)
    else:
        categorical_features = encoder.transform(categorical_features)
    
    # Combine features
    X = np.concatenate((numerical_features, categorical_features), axis=1)
    
    return X, scaler, encoder

# Streamlit app
st.set_page_config(
    page_title='AI Meal',
    page_icon='🧲️'
)
st.markdown("""
    <h1 style="text-align: center;">
        🍽️ <span style="background: linear-gradient(to right, red, orange, yellow, green, blue, indigo, violet);
        -webkit-background-clip: text; color: transparent;">AI-Based</span> 
    </h1>
    <h3 style="text-align: center; font-size: 22px; margin-bottom:60px">Meal Recommendation System</h3>
""", unsafe_allow_html=True)


# Sidebar for user input
st.sidebar.header("Nutrient Preferences")

# Add sliders for protein, carbs, and fat
protein = st.sidebar.slider("Protein (g)", 0, 100, 50)
carbs = st.sidebar.slider("Carbohydrates (g)", 0, 100, 30)
fat = st.sidebar.slider("Fat (g)", 0, 100, 20)

# Add toggle switches for dietary preferences
st.sidebar.header("Dietary Preferences")
include_meat = st.sidebar.checkbox("Include Meat", value=True)
include_dairy = st.sidebar.checkbox("Include Dairy", value=True)

# Function to calculate a custom score based on user preferences
def calculate_custom_score(row, protein_pref, carbs_pref, fat_pref):
    """
    Calculate a custom score based on how well the food item matches the user's preferences.
    """
    protein_diff = abs(row["Protein (g)"] - protein_pref)
    carbs_diff = abs(row["Carbohydrates (g)"] - carbs_pref)
    fat_diff = abs(row["Fat (g)"] - fat_pref)
    
    # Lower difference means better match
    total_diff = protein_diff + carbs_diff + fat_diff
    
    # Invert the difference to get a score (higher score = better match)
    score = 1 / (1 + total_diff)  # Add 1 to avoid division by zero
    return score

# Function to recommend food
def recommend_food(meal_type, protein_pref, carbs_pref, fat_pref, include_meat, include_dairy):
    """
    Recommend food items based on user preferences and meal type.
    """
    # Filter food items by meal type and create a copy
    meal_items = data[data["Meal_Type"] == meal_type].copy()  # Explicitly create a copy
    
    # Apply dietary preferences
    if not include_meat:
        meal_items = meal_items[meal_items["Category"] != "Meat"]
    if not include_dairy:
        meal_items = meal_items[meal_items["Category"] != "Dairy"]
    
    # Calculate custom scores based on user preferences
    meal_items["Custom_Score"] = meal_items.apply(
        lambda row: calculate_custom_score(row, protein_pref, carbs_pref, fat_pref),
        axis=1
    )
    
    # Sort by custom score and return top recommendations
    top_recommendations = meal_items.sort_values(by="Custom_Score", ascending=False).head(5)
    return top_recommendations[["Food_Item", "Category", "Calories (kcal)", "Protein (g)", "Carbohydrates (g)", "Fat (g)"]]

# Function to collect user feedback
def collect_feedback(user_id, food_item, meal_type, feedback):
    """
    Collect user feedback and save it to a file.
    """
    global feedback_data
    new_feedback = pd.DataFrame({
        "User_ID": [user_id],
        "Food_Item": [food_item],
        "Meal_Type": [meal_type],
        "Feedback": [feedback]
    })
    feedback_data = pd.concat([feedback_data, new_feedback], ignore_index=True)
    feedback_data.to_csv("feedback/user_feedback.csv", index=False)

# Function to update the model using reinforcement learning
def update_model_with_feedback():
    """
    Update the model based on user feedback using reinforcement learning.
    """
    global model, feedback_data, scaler, encoder
    if not feedback_data.empty:
        # Convert feedback to numerical values (1 for Like, 0 for Dislike)
        feedback_data["Feedback_Score"] = feedback_data["Feedback"].apply(lambda x: 1 if x == "Like" else 0)
        
        # Merge feedback with the main dataset
        merged_data = pd.merge(data, feedback_data, on=["Food_Item", "Meal_Type"], how="left")
        merged_data["Feedback_Score"] = merged_data["Feedback_Score"].fillna(0)  # Fill missing feedback with 0
        
        # Preprocess the data (fit the scaler and encoder on the new data)
        X, scaler, encoder = preprocess_data(merged_data, scaler, encoder, fit=True)
        y = merged_data["Feedback_Score"]
        
        # Retrain the model with feedback
        model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
        model.save("models/meal_recommendation_model.h5")

# Function to display recommendations in a table with Like/Dislike buttons
def display_recommendations_with_feedback(recommendations, meal_type):
    """
    Display recommendations in a table with Like/Dislike buttons.
    """
    if not recommendations.empty:
        # Round the values for Protein, Carbs, and Fat to 1 decimal point
        recommendations["Protein (g)"] = recommendations["Protein (g)"].apply(lambda x: f"{x:.1f}")
        recommendations["Carbohydrates (g)"] = recommendations["Carbohydrates (g)"].apply(lambda x: f"{x:.1f}")
        recommendations["Fat (g)"] = recommendations["Fat (g)"].apply(lambda x: f"{x:.1f}")
        
        # Display the table using Streamlit's native table function
        st.table(recommendations[["Food_Item", "Category", "Calories (kcal)", "Protein (g)", "Carbohydrates (g)", "Fat (g)"]])
        
        # Add Like/Dislike buttons for each recommendation in a single row
        for index, row in recommendations.iterrows():
            # Create a row for the food item and buttons
            col1, col2, col3 = st.columns([3, 1, 1])  # Adjust column widths as needed
            with col1:
                st.write(f"**{row['Food_Item']}**")  # Display the food item name
            with col2:
                if st.button(f"👍 Like", key=f"like_{meal_type}_{index}"):
                    collect_feedback("user1", row["Food_Item"], meal_type, "Like")
                    st.success(f"You liked {row['Food_Item']}!")
            with col3:
                if st.button(f"👎 Dislike", key=f"dislike_{meal_type}_{index}"):
                    collect_feedback("user1", row["Food_Item"], meal_type, "Dislike")
                    st.error(f"You disliked {row['Food_Item']}!")

# Display recommendations
# st.markdown("<h2 style='color: #FF5733;'>Recommended Meals</h2>", unsafe_allow_html=True)

# Breakfast
st.markdown("<h3 style='color: blue;'>🍳 Breakfast</h3>", unsafe_allow_html=True)
breakfast_recommendations = recommend_food("Breakfast", protein, carbs, fat, include_meat, include_dairy)
display_recommendations_with_feedback(breakfast_recommendations, "Breakfast")

# Lunch
st.markdown("<hr style='margin:50px'/>", unsafe_allow_html=True)
st.markdown("<h3 style='color: blue;'>🍲 Lunch</h3>", unsafe_allow_html=True)
lunch_recommendations = recommend_food("Lunch", protein, carbs, fat, include_meat, include_dairy)
display_recommendations_with_feedback(lunch_recommendations, "Lunch")

# Snack
st.markdown("<hr style='margin:50px'/>", unsafe_allow_html=True)
st.markdown("<h3 style='color: blue;'>🍪 Snack</h3>", unsafe_allow_html=True)
snack_recommendations = recommend_food("Snack", protein, carbs, fat, include_meat, include_dairy)
display_recommendations_with_feedback(snack_recommendations, "Snack")

# Dinner
st.markdown("<hr style='margin:50px'/>", unsafe_allow_html=True)
st.markdown("<h3 style='color: blue;'>🍽️ Dinner</h3>", unsafe_allow_html=True)
dinner_recommendations = recommend_food("Dinner", protein, carbs, fat, include_meat, include_dairy)
display_recommendations_with_feedback(dinner_recommendations, "Dinner")

# Update the model with feedback
if st.button("Update Recommendations with Feedback"):
    update_model_with_feedback()
    st.success("Model updated with your feedback!")