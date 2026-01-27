🌲 Forest Cover Type Prediction using Machine Learning
📌 Project Overview

This project predicts the forest cover type based on terrain and environmental features using machine learning classification techniques.
The complete workflow includes data analysis, feature engineering, model comparison, hyperparameter tuning, and deployment using a Streamlit web application for real-time prediction.

🎯 Problem Statement

To classify forest cover types using terrain and environmental attributes such as elevation, slope, hillshade, and distance-based features with high accuracy.

📊 Dataset Description

Dataset: Forest Cover Type Dataset

Total Records: 145,890

Features: 12 numerical features

Target Variable: Forest Cover Type (7 classes)

Forest Cover Classes

Spruce/Fir

Lodgepole Pine

Ponderosa Pine

Cottonwood/Willow

Aspen

Douglas-fir

Krummholz

🔍 Exploratory Data Analysis (EDA)

Checked dataset structure, missing values, and duplicates

Analyzed feature distributions using histograms

Detected outliers using boxplots

Studied feature relationships using correlation heatmaps

Analyzed class imbalance in the target variable

🛠️ Data Preprocessing & Feature Engineering

No missing values found

Outlier handling using statistical methods

Skewness correction for continuous features

Feature selection using correlation and feature importance

Label encoding applied to target variable

Feature alignment ensured during inference using saved feature list

🤖 Machine Learning Models Implemented

The following classification models were trained and evaluated:

Logistic Regression

Decision Tree

Random Forest

Extra Trees

Gradient Boosting

AdaBoost

XGBoost

📈 Model Evaluation

Models were compared using:

Accuracy

Confusion Matrix

Classification Report

Weighted F1-score

🏆 Best Performing Model

Random Forest Classifier

Accuracy: ~96%

Selected after hyperparameter tuning using RandomizedSearchCV

💾 Saved Model Artifacts

The following files are used for deployment:

label_encoder.pkl – Target label encoder

model_features.pkl – Feature alignment reference

🔗 Trained Model File

Due to GitHub’s file size limitations, the trained Random Forest model is hosted externally.

Download the model here:
👉 (Paste your Google Drive link here)

🌐 Streamlit Web Application

A Streamlit-based web application was developed to provide real-time predictions.

Application Features

Manual user input for all model features

Real-time forest cover type prediction

Prediction confidence (probability)

Top-3 class probability display

Clean and interactive UI

▶️ Run the Application
pip install -r requirements.txt
streamlit run app.py

🧪 Technologies Used

Python

Pandas, NumPy

Scikit-learn

XGBoost

Streamlit

Joblib

Matplotlib, Seaborn

📁 Project Structure
forest-cover-prediction/
│
├── app.py
├── README.md
├── requirements.txt
├── label_encoder.pkl
├── model_features.pkl
├── screenshots/
│   ├── ui_input.png
│   └── ui_prediction.png
└── .gitignore

📈 Results & Insights

Tree-based models significantly outperformed linear models

Elevation, hillshade, and distance features were the most influential

Random Forest provided the best balance between accuracy and generalization

🚀 Future Enhancements

Deploy the application on Streamlit Cloud

Improve handling of class imbalance

Add feature importance visualization to the UI

Integrate geospatial visualization

👨‍💻 Author

Aishwarya J
Forest Cover Type Prediction – Machine Learning Project 🌲
