# Gradio App for Loan Prediction

import gradio as gr
import pandas as pd
import pickle
import numpy as np

# 1. Load trained Logistic Regression pipeline
with open("loan_rf_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# 2. Prediction Function
def predict_loan(
    Gender, Married, Dependents, Education, Self_Employed,
    ApplicantIncome, CoapplicantIncome, LoanAmount,
    Loan_Amount_Term, Credit_History, Property_Area
):

    input_df = pd.DataFrame([{
        "Gender": Gender,
        "Married": Married,
        "Dependents": Dependents,
        "Education": Education,
        "Self_Employed": Self_Employed,
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": Credit_History,
        "Property_Area": Property_Area
    }])

    prediction = model.predict(input_df)[0]

    return "✅ Loan Approved" if prediction == 'Y' else "❌ Loan Not Approved"



# 3. Gradio Interface
inputs = [
    gr.Radio(["Male", "Female"], label="Gender"),
    gr.Radio(["Yes", "No"], label="Married"),
    gr.Dropdown(["0", "1", "2", "3+"], label="Dependents"),
    gr.Radio(["Graduate", "Not Graduate"], label="Education"),
    gr.Radio(["Yes", "No"], label="Self Employed"),
    gr.Number(label="Applicant Income"),
    gr.Number(label="Coapplicant Income"),
    gr.Number(label="Loan Amount"),
    gr.Number(label="Loan Amount Term", value=360),
    gr.Radio([1.0, 0.0], label="Credit History"),
    gr.Radio(["Urban", "Semiurban", "Rural"], label="Property Area")
]

app = gr.Interface(
    fn=predict_loan,
    inputs=inputs,
    outputs="text",
    title=" Loan Approval Prediction System",
    
)

app.launch(share=True)
