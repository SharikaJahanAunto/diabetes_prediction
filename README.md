🩺 Diabetes Prediction using Machine Learning

This project implements an end-to-end Machine Learning pipeline to predict diabetes based on medical attributes. It covers data preprocessing, model training, evaluation, and deployment with a user-friendly web interface.

📌 Project Overview

The goal of this project is to build a reliable binary classification system that predicts whether a patient is diabetic or not using clinical features such as glucose level, BMI, blood pressure, insulin level, and age. The solution follows standard Machine Learning practices and is deployed as a web application.

📊 Dataset

Name: Diabetes Dataset

Type: Structured tabular data

Target Variable: Outcome (0 = Not Diabetic, 1 = Diabetic)

Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age

⚙️ Methodology

The project follows these key steps:

Data loading and validation

Data preprocessing (handling missing values, scaling, feature preparation)

Machine Learning pipeline creation

Model training using Logistic Regression

Cross-validation for robustness evaluation

Hyperparameter tuning using Grid Search

Final model selection and test evaluation

Web interface development using Gradio

Deployment on Hugging Face Spaces

🧠 Model Used

Algorithm: Logistic Regression

Reason: Efficient, interpretable, and well-suited for binary classification problems, especially in healthcare applications.

📈 Model Evaluation

The model is evaluated using:

Accuracy

Precision, Recall, and F1-score

Confusion Matrix

Cross-validation mean accuracy and standard deviation

🌐 Web Application

A Gradio-based web interface allows users to input patient medical data and receive real-time diabetes predictions.
The application is deployed publicly using Hugging Face Spaces.

🛠️ Technologies Used

Python

Pandas, NumPy

Scikit-learn

Gradio

Hugging Face Spaces

📁 Project Structure
├── app.py
├── diabetes.csv
├── requirements.txt
├── README.md

▶️ How to Run Locally
pip install -r requirements.txt
python app.py

📄 License

This project is developed for academic purposes.
