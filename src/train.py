"""Train, evaluate, and save the churn prediction model."""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:  # Works both as `python src/train.py` and when imported by the app.
    from .data_generation import save_dataset
except ImportError:
    from data_generation import save_dataset


TARGET, ID_COLUMN, RANDOM_STATE = "churn", "customer_id", 42


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Keep preparation in one pipeline so the test set cannot leak into training."""
    numeric_steps = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    category_steps = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numeric", numeric_steps, numeric), ("categorical", category_steps, categorical)])


def metrics_for(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, predictions), 3),
        "precision": round(precision_score(y_test, predictions), 3),
        "recall": round(recall_score(y_test, predictions), 3),
        "roc_auc": round(roc_auc_score(y_test, probabilities), 3),
    }


def train_and_save(project_root: Path) -> dict:
    """Generate data when needed, compare models, and save the better model."""
    data_path = project_root / "data" / "customer_churn.csv"
    if not data_path.exists():
        save_dataset(data_path)
    data = pd.read_csv(data_path)
    x = data.drop(columns=[TARGET, ID_COLUMN])
    y = data[TARGET].map({"Yes": 1, "No": 0})
    numeric = x.select_dtypes(include="number").columns.tolist()
    categorical = x.select_dtypes(exclude="number").columns.tolist()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=350, min_samples_leaf=4, class_weight="balanced", random_state=RANDOM_STATE),
    }
    results, fitted = {}, {}
    for name, estimator in candidates.items():
        model = Pipeline([("preprocessor", build_preprocessor(numeric, categorical)), ("model", estimator)])
        model.fit(x_train, y_train)
        results[name], fitted[name] = metrics_for(model, x_test, y_test), model

    # In retention work, missing a customer who leaves is costly, so maximise recall first.
    best_name = max(results, key=lambda name: (results[name]["recall"], results[name]["roc_auc"]))
    best_model = fitted[best_name]
    model_dir = project_root / "models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(best_model, model_dir / "churn_model.joblib")
    feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
    estimator = best_model.named_steps["model"]
    # Trees expose impurity importance; the linear baseline exposes coefficients.
    importances = estimator.feature_importances_ if hasattr(estimator, "feature_importances_") else abs(estimator.coef_[0])
    importance = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).head(12).to_dict(orient="records")
    summary = {"selected_model": best_name, "metrics": results, "feature_importance": importance}
    (model_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    summary = train_and_save(Path(__file__).resolve().parents[1])
    print(f"Selected model: {summary['selected_model']}")
    print(summary["metrics"][summary["selected_model"]])
