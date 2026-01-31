import gradio as gr
import numpy as np
import pandas as pd
import pickle

# Load trained model
with open("diabetes_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# Prediction function
def predict_diabetes(
    Pregnancies,
    Glucose,
    BloodPressure,
    SkinThickness,
    Insulin,
    BMI,
    DiabetesPedigreeFunction,
    Age
):
    # Create DataFrame with correct column order
    input_data = pd.DataFrame([{
        "Pregnancies": Pregnancies,
        "Glucose": Glucose,
        "BloodPressure": BloodPressure,
        "SkinThickness": SkinThickness,
        "Insulin": Insulin,
        "BMI": BMI,
        "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
        "Age": Age
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    result = "Diabetic" if prediction == 1 else "Not Diabetic"

    return f"{result} (Risk Probability: {probability:.2%})"


# Gradio Interface
app = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies", value=2),
        gr.Number(label="Glucose", value=120),
        gr.Number(label="Blood Pressure", value=70),
        gr.Number(label="Skin Thickness", value=20),
        gr.Number(label="Insulin", value=80),
        gr.Number(label="BMI", value=25.0),
        gr.Number(label="Diabetes Pedigree Function", value=0.5),
        gr.Number(label="Age", value=30),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Diabetes Prediction App",
    description="Predict whether a person is diabetic using medical parameters."
)

if __name__ == "__main__":
    app.launch()
