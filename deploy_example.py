
import pickle
import numpy as np

# 1. Load the scaler and the model
with open('scaler.pkl', 'rb') as sf:
    scaler = pickle.load(sf)

with open('best_knn_model.pkl', 'rb') as mf:
    model = pickle.load(mf)

# 2. Example new patient data (matching original feature columns format)
# Age, BMI, Glucose, Insulin, HOMA, Leptin, Adiponectin, Resistin, MCP-1
new_patient_data = np.array([[48, 23.5, 70, 2.707, 0.467409, 8.8071, 9.7024, 7.99585, 417.114]])

# 3. Preprocess and Scale the data
new_patient_scaled = scaler.transform(new_patient_data)

# 4. Predict
prediction = model.predict(new_patient_scaled)
print(f"Predicted Class: {prediction[0]} (1 = Healthy Control, 2 = Patient)")
