from pathlib import Path

from src.data_generation import generate_customer_data
from src.train import train_and_save


def test_generated_data_has_expected_columns():
    data = generate_customer_data(20)
    assert len(data) == 20
    assert {"customer_id", "churn", "contract", "monthly_charges"}.issubset(data.columns)


def test_training_creates_model(tmp_path: Path):
    summary = train_and_save(tmp_path)
    assert (tmp_path / "models" / "churn_model.joblib").exists()
    assert summary["selected_model"] in {"Logistic Regression", "Random Forest"}
