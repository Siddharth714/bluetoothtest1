import joblib
import pandas as pd

# Load the combined object (model, scaler, and feature list)
combined = joblib.load("combined_model_scaler_smoteenn_randomized.pkl")
model = combined["model"]
scaler = combined["scaler"]
selected_features = combined["features"]

one_hot_mapping = {
    3: [0, 0, 0, 1],  # Moderate Demented
    1: [0, 1, 0, 0],  # Very Mild Demented
    2: [0, 0, 1, 0],  # Mild Demented
    0: [1, 0, 0, 0]   # Healthy
}

# Define mapping from numeric labels to category names.
# Adjust these mappings to match your labeling scheme.
label_to_category = {
    0: "Moderate Demented",
    1: "Very Mild Demented",
    2: "Mild Demented",
    3: "Healthy"
}

def predict_from_excel(excel_file: str) -> str:
    """
    Reads an Excel file containing one row with the averaged feature values,
    applies the saved scaler, and predicts the category using the trained model.
    """
    # Load the Excel file (should contain one row of data with headers matching selected_features)
    input_df = pd.read_csv(excel_file)
    
    # Ensure the DataFrame has exactly the selected features in the correct order
    input_df = input_df[selected_features]
    
    # Scale the input data using the saved scaler (resulting in a NumPy array)
    input_scaled = scaler.transform(input_df)
    
    # Predict the label using the model (using the scaled NumPy array)
    ##predicted_label = model.predict(input_scaled)[0]
    pred_labels = model.predict(input_scaled)[0]
    print(pred_labels)
    print(one_hot_mapping[pred_labels])
    
    
    # Map the numeric prediction to the corresponding category name
    return label_to_category.get(pred_labels, "Unknown Category")
    

# Example usage:
excel_filename = "input_data.csv"  # This file should contain one row with columns matching 'selected_features'
predicted_category = predict_from_excel(excel_filename)
print(f"The predicted category is: {predicted_category}")
