import joblib
import pandas as pd

# Load the combined object (model, scaler, and feature list)
combined = joblib.load("combined_model_scaler.pkl")
model = combined["model"]
scaler = combined["scaler"]
selected_features = combined["features"]

# Mapping from label to category name
label_to_category = {
    3: "Healthy",
    1: "Very Mild Demented",
    2: "Mild Demented",
    0: "Moderate Demented"
}

def predict_from_excel(excel_file: str) -> str:
    """
    Reads an Excel file containing one row with the averaged feature values,
    preprocesses the data, and predicts the category.
    """
    # Load the Excel file (if you are using an Excel file instead of CSV)
    input_df = pd.read_csv(excel_file)
    
    # Ensure the DataFrame has exactly the selected features in the correct order
    input_df = input_df[selected_features]
    
    # Standardize the input data using the loaded scaler
    input_scaled = scaler.transform(input_df)
    
    # Predict using the NumPy array (no feature names)
    predicted_label = model.predict(input_scaled)[0]
    
    return label_to_category.get(predicted_label, "Unknown Category")

# Example usage:
excel_filename = "input_data.csv"  # Ensure this file has one row and the correct column headers
predicted_category = predict_from_excel(excel_filename)
print(f"The predicted category is: {predicted_category}")
