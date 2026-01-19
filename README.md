# 🩺 Diabetes Prediction using Machine Learning

An end-to-end Machine Learning project that predicts the likelihood of diabetes using clinical data. The project demonstrates the complete ML lifecycle—from data preprocessing and model training to deployment as a web application.

---

## 🚀 Overview

This project builds a binary classification system to determine whether a patient is diabetic based on medical attributes such as glucose level, BMI, blood pressure, insulin, and age. A standardized ML pipeline is used to ensure reproducibility, robustness, and scalability. The final model is deployed with an interactive Gradio interface on Hugging Face Spaces.

---

## 📊 Dataset

* **Type:** Tabular medical dataset
* **Target:** `Outcome` (0 = Not Diabetic, 1 = Diabetic)
* **Features:**

  * Pregnancies
  * Glucose
  * BloodPressure
  * SkinThickness
  * Insulin
  * BMI
  * DiabetesPedigreeFunction
  * Age

---

## ⚙️ Workflow

1. Data loading and validation
2. Data preprocessing (missing value handling, scaling)
3. Machine Learning pipeline construction
4. Model training using Logistic Regression
5. Cross-validation for performance stability
6. Hyperparameter tuning with Grid Search
7. Final model evaluation on test data
8. Web application development with Gradio
9. Deployment on Hugging Face Spaces

---

## 🧠 Model

* **Algorithm:** Logistic Regression
* **Why:** Simple, interpretable, efficient, and well-suited for binary classification problems in healthcare.

---

## 📈 Evaluation Metrics

* Accuracy
* Precision, Recall, F1-score
* Confusion Matrix
* Cross-validation mean score and standard deviation

---

## 🌐 Web Application

The project includes a Gradio-based web interface that allows users to input patient data and receive real-time diabetes predictions.
The application is publicly accessible via **Hugging Face Spaces**.

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Gradio
* Hugging Face Spaces

---

## 📁 Project Structure

```
├── app.py
├── diabetes.csv
├── requirements.txt
├── README.md
```

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
python app.py
```

---

## 📄 License

This project is intended for educational and academic use.

