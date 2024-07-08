import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model



# Load the trained model
model = load_model("landslide_detection_model.h5")

# Load the new data for prediction
new_data = pd.read_excel("new_data.xlsx")

# Preprocess the new data
scaler = StandardScaler()
X_new = scaler.fit_transform(new_data.values)

# Make predictions
predictions = model.predict(X_new)

# Convert predictions to binary classes (0 for no landslide, 1 for landslide)
binary_predictions = (predictions > 0.5).astype(int)

# Print the predictions
for i, prediction in enumerate(binary_predictions):
    if prediction == 1:
        print(f"Data point {i+1}: Landslide is likely to occur")
    else:
        print(f"Data point {i+1}: No landslide is likely to occur")