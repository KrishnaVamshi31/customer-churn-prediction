"""Create a realistic, reproducible demo dataset for the churn project."""

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_STATE = 42


def generate_customer_data(n_customers: int = 2_500, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Return synthetic telecom-style customer data with a churn label."""
    rng = np.random.default_rng(random_state)
    tenure = rng.integers(1, 73, n_customers)
    monthly_charges = np.clip(rng.normal(67, 24, n_customers), 18, 125).round(2)
    total_charges = (monthly_charges * tenure * rng.uniform(0.92, 1.08, n_customers)).round(2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n_customers, p=[0.56, 0.24, 0.20])
    internet_service = rng.choice(["Fiber optic", "DSL", "No"], n_customers, p=[0.45, 0.38, 0.17])
    payment_method = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n_customers, p=[0.35, 0.18, 0.24, 0.23])
    tech_support = rng.choice(["Yes", "No"], n_customers, p=[0.34, 0.66])
    online_security = rng.choice(["Yes", "No"], n_customers, p=[0.31, 0.69])
    senior_citizen = rng.choice(["Yes", "No"], n_customers, p=[0.16, 0.84])

    # These rules reflect plausible churn drivers but include random variation.
    log_odds = -1.65
    log_odds += 1.25 * (contract == "Month-to-month") + 0.30 * (contract == "One year")
    log_odds += 0.70 * (internet_service == "Fiber optic")
    log_odds += 0.48 * (tech_support == "No") + 0.30 * (online_security == "No")
    log_odds += 0.40 * (payment_method == "Electronic check")
    log_odds += 0.24 * (senior_citizen == "Yes") + 0.018 * (monthly_charges - 65) - 0.025 * tenure
    log_odds += rng.normal(0, 0.65, n_customers)
    churn = np.where(rng.random(n_customers) < 1 / (1 + np.exp(-log_odds)), "Yes", "No")
    return pd.DataFrame({
        "customer_id": [f"CUST-{i:05d}" for i in range(1, n_customers + 1)],
        "tenure_months": tenure, "monthly_charges": monthly_charges, "total_charges": total_charges,
        "contract": contract, "internet_service": internet_service, "payment_method": payment_method,
        "tech_support": tech_support, "online_security": online_security,
        "senior_citizen": senior_citizen, "churn": churn,
    })


def save_dataset(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_customer_data().to_csv(path, index=False)
    return path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(f"Dataset saved to {save_dataset(root / 'data' / 'customer_churn.csv')}")
