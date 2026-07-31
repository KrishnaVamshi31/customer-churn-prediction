# ChurnGuard — Customer Churn Prediction

An end-to-end machine-learning project that predicts which subscription customers may leave, allowing a business to offer timely retention support.

## Why this project matters

Keeping a customer is typically less expensive than acquiring a new one. This classifier helps a subscription business prioritise customers for proactive outreach. It is a **decision-support tool**, not an automatic decision maker.

## What is included

- Reproducible telecom-style demo data generation
- scikit-learn preprocessing pipeline: imputation, scaling, and categorical encoding
- Logistic Regression vs Random Forest comparison
- Evaluation using accuracy, precision, recall, and ROC-AUC
- Feature-importance visualisation
- Streamlit dashboard for live predictions and insights
- Basic automated tests

## Structure

```
Customer-Churn-Prediction/
├── app.py                 # Streamlit user interface
├── src/
│   ├── data_generation.py # Creates reproducible demo data
│   └── train.py           # Trains, evaluates, and saves the model
├── tests/                 # Automated checks
├── data/                  # Generated during the first training run
└── models/                # Saved model and performance summary
```

## Run locally

```powershell
cd $HOME\Documents\Customer-Churn-Prediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\train.py
streamlit run app.py
```

## ML workflow, explained simply

1. **Generate/load data.** The supplied dataset is synthetic, so the project runs immediately and reproducibly. For a real deployment, replace it with approved real customer data.
2. **Split the data.** 80% trains the models and 20% remains unseen until evaluation.
3. **Prepare data inside a pipeline.** Numeric values are imputed and scaled; text categories are encoded. This avoids data leakage from the test set.
4. **Compare models.** Logistic Regression is a simple, explainable baseline. Random Forest captures more complex relationships.
5. **Select a model.** The project prioritises recall, then ROC-AUC, because missing a likely churner loses a retention opportunity.
6. **Deploy.** The selected model is saved with `joblib`; the Streamlit app loads it for live predictions.

## Metrics

| Metric | What it tells you |
|---|---|
| Accuracy | How often the model was correct overall. |
| Precision | Of customers flagged as churners, how many really churned? |
| Recall | Of customers who churned, how many did we catch? |
| ROC-AUC | How well the model ranks churn risk across thresholds. |

## Interview talking points

- "I kept preprocessing in a scikit-learn Pipeline, so transformations are learned from the training data only."
- "I used Logistic Regression as a baseline and Random Forest for non-linear patterns, then compared them fairly on the same test set."
- "I prioritised recall because missed churners are missed retention opportunities."
- "Feature importance supports investigation, but it does not prove causation."
- "Before real deployment, I would validate on a representative dataset and monitor performance drift."

## Resume bullet

> Built ChurnGuard, an end-to-end customer churn prediction application using Python, scikit-learn, and Streamlit; compared classification models, evaluated recall and ROC-AUC, and delivered live risk predictions with retention recommendations.
