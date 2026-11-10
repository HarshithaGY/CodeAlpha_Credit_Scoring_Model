# 💳 CreditLens – Credit Risk Prediction System

### CodeAlpha Machine Learning Internship Project

## 📌 Project Overview

CreditLens is a Machine Learning-based Credit Risk Prediction System designed to predict whether a loan applicant has good or bad credit risk based on their financial and personal information.

The project uses machine learning classification algorithms to analyze applicant data and support credit risk assessment through an interactive web application built using Streamlit.

## 🎯 Project Objectives

- Analyze customer financial information.
- Predict credit risk using machine learning.
- Compare different classification algorithms.
- Evaluate model performance using standard metrics.
- Build an interactive and user-friendly prediction interface.

## 📊 Dataset

**Dataset:** German Credit Risk Dataset  
**Source:** UCI Machine Learning Repository

- Total records: 1,000
- Input features: 20
- Target classes:
  - Good Credit (1)
  - Bad Credit (0)

## 🤖 Machine Learning Algorithms

Three classification algorithms were implemented and compared:

| Algorithm | Accuracy |
|---|---:|
| Logistic Regression | 70.5% |
| Decision Tree | 67.5% |
| Random Forest | 74.5% |

**Best-performing model:** Random Forest

*These accuracy values are from the initial model comparison on the test dataset.*

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- VS Code

## ⚙️ Project Workflow

1. Dataset collection and loading.
2. Data cleaning and preprocessing.
3. Exploratory Data Analysis (EDA).
4. Feature scaling and encoding.
5. Model training and comparison.
6. Model evaluation.
7. Saving the trained model.
8. Developing the interactive Streamlit application.

## ✨ Application Features

- Interactive credit risk prediction.
- User-friendly input form.
- Good credit and bad credit prediction.
- Prediction probability display.
- Model performance analytics.
- Confusion matrix and classification report.
- Methodology overview.

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

### 2. Open the project folder

```bash
cd CodeAlpha_Credit_Scoring_Model
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🎥 Project Demonstration

A demonstration video of the CreditLens application is included in this repository.

The video showcases the interface, credit risk prediction, and model analytics.

## 📚 Key Learnings

- Understanding the machine learning classification workflow.
- Data preprocessing and feature engineering.
- Training and comparing multiple ML algorithms.
- Evaluating classification models.
- Building interactive applications using Streamlit.
- Integrating machine learning models into a real-time interface.

## ⚠️ Disclaimer

This project is developed for educational purposes as part of the CodeAlpha Machine Learning Internship. Predictions are based on the selected dataset and trained model. They should not be used as the sole basis for real-world financial or loan approval decisions.

## 👩‍💻 Internship

**Organization:** CodeAlpha  
**Domain:** Machine Learning  
**Project:** Credit Scoring Model

---

⭐ If you find this project interesting, feel free to explore the repository!
