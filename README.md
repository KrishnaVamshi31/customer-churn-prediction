# 📉 Customer Churn Prediction

> An end-to-end Machine Learning application that predicts customer churn using Scikit-learn Pipelines, Random Forest, Logistic Regression, and an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📌 Overview

Customer churn is one of the most critical business problems in subscription-based industries. Identifying customers who are likely to leave enables companies to take proactive retention measures.

This project implements a complete Machine Learning workflow—from data generation and preprocessing to model training, evaluation, persistence, and deployment through a Streamlit web application.

---

## ✨ Features

- End-to-end Machine Learning pipeline
- Synthetic customer dataset generation
- Data preprocessing using Scikit-learn Pipelines
- Missing value imputation
- Feature scaling
- One-Hot Encoding for categorical variables
- Logistic Regression baseline model
- Random Forest classifier
- Automated model comparison
- Model persistence using Joblib
- Interactive Streamlit dashboard
- Performance visualization
- Real-time churn prediction

---

# 🏗 Project Architecture

```

Customer Dataset
│
▼
Data Preprocessing
│
├── Missing Value Handling
├── Feature Scaling
└── One-Hot Encoding
│
▼
Train / Test Split
│
▼
Model Training
├── Logistic Regression
└── Random Forest
│
▼
Model Evaluation
│
▼
Best Model Selection
│
▼
Model Serialization (.joblib)
│
▼
Streamlit Dashboard
│
▼
Customer Churn Prediction

```

---

# 📂 Project Structure

```

Customer-Churn-Prediction/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   └── customer_churn.csv
│
├── models/
│   ├── churn_model.joblib
│   └── metrics.json
│
├── src/
│   ├── data_generation.py
│   └── train.py
│
└── tests/

```

---

# 🛠 Tech Stack

| Category | Technologies |
|-----------|--------------|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Model Persistence | Joblib |
| Visualization | Plotly |
| Frontend | Streamlit |
| Testing | PyTest |

---

# 📊 Dataset

The project uses a **synthetically generated customer churn dataset** consisting of approximately **2,500 customer records**.

Each customer contains features such as:

- Customer Tenure
- Monthly Charges
- Total Charges
- Contract Type
- Internet Service
- Payment Method
- Online Security
- Tech Support
- Senior Citizen Status
- Churn (Target Variable)

The dataset is generated using business-inspired rules to simulate realistic customer behavior.

---

# ⚙ Machine Learning Pipeline

### Numerical Features

- Median Imputation
- Standard Scaling

### Categorical Features

- Most Frequent Imputation
- One-Hot Encoding

Both pipelines are combined using Scikit-learn's **ColumnTransformer** to ensure consistent preprocessing during both training and inference.

---

# 🤖 Models Used

## Logistic Regression

- Baseline classification model
- Balanced class weights
- Max Iterations = 1000

---

## Random Forest

- 350 Decision Trees
- Balanced class weights
- Minimum samples per leaf = 4

The best-performing model is automatically selected and saved for deployment.

---

# 📈 Model Performance

| Metric | Logistic Regression | Random Forest |
|---------|--------------------:|--------------:|
| Accuracy | 66.2% | 66.6% |
| Precision | 53.6% | 54.2% |
| Recall | 68.3% | 65.6% |
| ROC-AUC | 71.6% | 71.1% |

Model selection prioritizes **Recall** to better identify customers at risk of churning.

---

# 💻 Streamlit Dashboard

The web application allows users to:

- Enter customer information
- Predict churn likelihood
- View model metrics
- Explore performance visualizations

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/KrishnaVamshi31/customer-churn-prediction.git
```

Navigate into the project

```bash
cd customer-churn-prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Launch the Streamlit application

```bash
streamlit run app.py
```

---

# 📷 Screenshots

## Dashboard

```
Add dashboard screenshot here
```

---

## Prediction Result

```
Add prediction screenshot here
```

---

## Performance Metrics

```
Add metrics screenshot here
```

---

# 🔮 Future Improvements

- Train on a real-world telecom churn dataset
- Hyperparameter tuning using GridSearchCV
- SHAP-based model explainability
- Cross-validation
- Model deployment on Streamlit Community Cloud
- Docker containerization
- CI/CD pipeline using GitHub Actions
- REST API using FastAPI

---

# 🎯 Learning Outcomes

This project demonstrates practical experience in:

- Machine Learning
- Binary Classification
- Data Preprocessing
- Feature Engineering
- Model Evaluation
- Scikit-learn Pipelines
- Streamlit Development
- Model Serialization
- Software Engineering Best Practices

---

# 🏷 Repository Topics

```
machine-learning
customer-churn
classification
python
scikit-learn
streamlit
predictive-analytics
data-science
```

---

# 👨‍💻 Author

**Krishna Vamshi**

GitHub: https://github.com/KrishnaVamshi31

---

# 📄 License

This project is licensed under the MIT License.
