"""Interactive Streamlit dashboard for ChurnGuard."""

from pathlib import Path
import json

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.train import train_and_save


st.set_page_config(page_title="ChurnGuard", page_icon="📉", layout="wide")
ROOT = Path(__file__).resolve().parent
MODEL_PATH, METRICS_PATH = ROOT / "models" / "churn_model.joblib", ROOT / "models" / "metrics.json"


@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        train_and_save(ROOT)
    return joblib.load(MODEL_PATH), json.loads(METRICS_PATH.read_text(encoding="utf-8"))


model, summary = load_assets()
st.title("ChurnGuard · Customer Churn Prediction")
st.caption("Identify subscription customers who may leave—and act before they do.")
predict_tab, insights_tab, about_tab = st.tabs(["Predict risk", "Model insights", "How it works"])

with predict_tab:
    st.subheader("Customer profile")
    left, right = st.columns(2)
    with left:
        tenure = st.slider("Tenure (months)", 1, 72, 12)
        monthly = st.slider("Monthly charges ($)", 18.0, 125.0, 75.0, 1.0)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    with right:
        payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
        support = st.selectbox("Tech support", ["No", "Yes"])
        security = st.selectbox("Online security", ["No", "Yes"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
    customer = pd.DataFrame([{"tenure_months": tenure, "monthly_charges": monthly, "total_charges": round(tenure * monthly, 2), "contract": contract, "internet_service": internet, "payment_method": payment, "tech_support": support, "online_security": security, "senior_citizen": senior}])
    probability = model.predict_proba(customer)[0, 1]
    risk = "High" if probability >= 0.60 else "Medium" if probability >= 0.35 else "Low"
    a, b = st.columns([1, 2])
    with a:
        st.metric("Churn probability", f"{probability:.1%}")
        st.metric("Risk level", risk)
    with b:
        messages = {
            "High": (st.error, "Contact this customer with a tailored retention offer and support check-in."),
            "Medium": (st.warning, "Send a value-focused plan or loyalty offer."),
            "Low": (st.success, "Maintain engagement; no urgent intervention is indicated."),
        }
        show, text = messages[risk]
        show(f"Recommended action: {text}")

with insights_tab:
    selected = summary["selected_model"]
    st.subheader(f"Selected model: {selected}")
    metrics = summary["metrics"][selected]
    for column, (label, key) in zip(st.columns(4), [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("ROC-AUC", "roc_auc")]):
        column.metric(label, f"{metrics[key]:.1%}")
    st.info("Recall matters here because a missed churner is a missed retention opportunity.")
    features = pd.DataFrame(summary["feature_importance"]).sort_values("importance")
    st.plotly_chart(px.bar(features, x="importance", y="feature", orientation="h", title="Most influential model features"), use_container_width=True)
    st.caption("Feature importance shows model reliance, not cause and effect.")

with about_tab:
    st.subheader("What this project demonstrates")
    st.markdown("""
    - **Leakage-safe preprocessing:** scaling and one-hot encoding occur inside a scikit-learn pipeline.
    - **Model selection:** Logistic Regression and Random Forest are tested on the same unseen test set.
    - **Business evaluation:** recall, precision, accuracy, and ROC-AUC guide the decision.
    - **Responsible use:** predictions prioritise helpful outreach; they should not make irreversible decisions.
    """)
    st.caption("The included dataset is synthetic and reproducible for demonstration. Replace it with approved real data before business use.")
