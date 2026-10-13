"""# 🎗️ Breast Cancer Classification Web App

This repository contains a machine learning web application built with **Streamlit** that classifies clinical samples into **Healthy Controls** or **Breast Cancer Patients** based on 9 specific biomarkers. 

Our optimized **K-Nearest Neighbors (KNN)** model achieves an accuracy of **83.33%** on the holdout test set.

## 📊 Clinical Features
The model utilizes the following 9 bio-measurements:
- **Age**: Age of the individual (years)
- **BMI**: Body Mass Index (kg/m²)
- **Glucose**: Serum glucose levels (mg/dL)
- **Insulin**: Serum insulin levels (µU/mL)
- **HOMA-IR**: Homeostatic Model Assessment of Insulin Resistance
- **Leptin**: Leptin hormone levels (ng/mL)
- **Adiponectin**: Adiponectin protein levels (µg/mL)
- **Resistin**: Resistin protein levels (ng/mL)
- **MCP-1**: Monocyte Chemoattractant Protein-1 levels (pg/mL)

## 📂 Repository Contents
- `app.py`: The main Streamlit web application script.
- `best_knn_model.pkl`: The serialized trained K-Nearest Neighbors classifier.
- `scaler.pkl`: The fitted StandardScaler used to normalize patient features before classification.
- `deploy_example.py`: A simplified command-line template for running offline inferences.
- `datar2.csv`: The original clinical dataset used to train and test the model.

---
