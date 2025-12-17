# Credit Card Fraud Detection (ML)

End-to-end machine learning project on the Kaggle credit card fraud dataset. The goal is to detect fraudulent transactions using multiple models and deploy the best model as a simple Streamlit app.[web:41]

## Features

- Data preprocessing, EDA and class imbalance analysis  
- Models: Logistic Regression, SVM, Random Forest (with class_weight='balanced')  
- Evaluation using precision, recall, F1-score and confusion matrix for the fraud class  
- Model comparison plot and final model selection (Random Forest)  
- Streamlit web app where users enter transaction amount and intuitive risk scores to get fraud probability.

## Files

- `fraud_detection_project.ipynb` – main notebook with EDA and model training  
- `streamlit.py` – Streamlit app script  
- `requirements.txt` – Python dependencies  

## How to run the app

pip install -r requirements.txt
streamlit run streamlit.py
