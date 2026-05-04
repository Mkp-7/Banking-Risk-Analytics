"""
Banking Operational Risk ML Engine
- Supervised: Logistic Regression + Random Forest → predict bank failure risk
- Unsupervised: K-Means → cluster banks by risk profile
- Outputs: model artifacts, evaluation metrics, feature importance
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle

DB_PATH = os.path.join(os.path.dirname(__file__), "../../data/banking_risk.db")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../../data/models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "ROA", "ROE", "capital_adequacy_ratio", "npl_ratio",
    "loan_to_deposit_ratio", "net_interest_margin", "cost_to_income_ratio"
]


# ── Load Data ─────────────────────────────────────────────────────────────────

def load_data():
    conn = sqlite3.connect(DB_PATH)
    institutions = pd.read_sql("SELECT * FROM institutions", conn)
    failures = pd.read_sql("SELECT * FROM failures", conn)
    conn.close()
    print(f"  institutions columns: {list(institutions.columns)}")
    print(f"  failures columns:     {list(failures.columns)}")
    return institutions, failures


def find_col(df, candidates):
    """Return first matching column from candidates list, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ── Prepare Dataset ───────────────────────────────────────────────────────────

def prepare_supervised_dataset(institutions: pd.DataFrame, failures: pd.DataFrame) -> pd.DataFrame:
    df = institutions.copy()

    # Convert all feature columns to numeric
    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Detect cert column in institutions
    inst_cert_col = find_col(df, ["cert_id", "CERT", "cert", "QBFCERT"])
    if inst_cert_col is None:
        raise ValueError(f"No cert column found. Columns: {list(df.columns)}")
    df["cert_str"] = df[inst_cert_col].astype(str)

    # Build failed cert set from failures table
    failed_certs = set()
    for col in ["CERT", "cert_id", "cert", "QBFCERT"]:
        if col in failures.columns:
            failed_certs.update(failures[col].dropna().astype(str).tolist())
    print(f"  Failed cert set size: {len(failed_certs)}")

    # Binary label
    df["failed_historically"] = df["cert_str"].isin(failed_certs).astype(int)
    df["high_risk_kri"] = (
        (pd.to_numeric(df.get("ROA"), errors="coerce") < 0) |
        (pd.to_numeric(df.get("capital_adequacy_ratio"), errors="coerce") < 8) |
        (pd.to_numeric(df.get("npl_ratio"), errors="coerce") > 10)
    ).astype(int)
    df["at_risk"] = ((df["failed_historically"] == 1) | (df["high_risk_kri"] == 1)).astype(int)

    # Detect label columns flexibly and rename to standard names
    name_col  = find_col(df, ["bank_name", "INSTNAME", "name", "NAME"])
    state_col = find_col(df, ["state", "STNAME", "STATE", "stname"])
    size_col  = find_col(df, ["size_bucket", "SIZE_BUCKET", "size"])

    rename_map = {}
    meta_cols = ["at_risk"]
    if name_col:
        meta_cols.append(name_col)
        rename_map[name_col] = "bank_name"
    if state_col:
        meta_cols.append(state_col)
        rename_map[state_col] = "state"
    if size_col:
        meta_cols.append(size_col)
        rename_map[size_col] = "size_bucket"

    available_features = [f for f in FEATURES if f in df.columns]
    df_model = df[available_features + meta_cols].copy().rename(columns=rename_map)

    # Fill any missing FEATURES columns with NaN
    for f in FEATURES:
        if f not in df_model.columns:
            df_model[f] = np.nan

    df_model = df_model.dropna(subset=available_features, thresh=max(3, len(available_features) // 2))

    print(f"  Dataset: {len(df_model):,} banks | At-risk: {df_model['at_risk'].sum():,} ({df_model['at_risk'].mean()*100:.1f}%)")
    return df_model


# ── Supervised Models ─────────────────────────────────────────────────────────

def train_supervised(df: pd.DataFrame) -> dict:
    X = df[FEATURES]
    y = df["at_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=-1
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
            ))
        ])
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")

        metrics = {
            "accuracy":    round(accuracy_score(y_test, y_pred), 4),
            "precision":   round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":      round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1":          round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc":     round(roc_auc_score(y_test, y_prob), 4),
            "cv_auc_mean": round(cv_scores.mean(), 4),
            "cv_auc_std":  round(cv_scores.std(), 4),
        }
        results[name] = {"pipeline": pipe, "metrics": metrics}
        print(f"  {name}: AUC={metrics['roc_auc']} | Acc={metrics['accuracy']} | F1={metrics['f1']}")

    # Feature importance from Random Forest
    rf_model = results["Random Forest"]["pipeline"].named_steps["model"]
    importance_df = pd.DataFrame({
        "feature": FEATURES,
        "importance": rf_model.feature_importances_
    }).sort_values("importance", ascending=False)
    results["feature_importance"] = importance_df

    # Save best model
    best_name = max(
        [k for k in results if k != "feature_importance"],
        key=lambda k: results[k]["metrics"]["roc_auc"]
    )
    with open(os.path.join(MODEL_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump(results[best_name]["pipeline"], f)

    metrics_out = {k: v["metrics"] for k, v in results.items() if k != "feature_importance"}
    with open(os.path.join(MODEL_DIR, "model_metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"\n  Best model: {best_name} (AUC={results[best_name]['metrics']['roc_auc']})")
    return results


# ── K-Means Clustering ────────────────────────────────────────────────────────

def train_clustering(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df = df.copy()
    df["cluster"] = kmeans.fit_predict(X_scaled)

    cluster_summary = df.groupby("cluster")[["ROA", "capital_adequacy_ratio", "npl_ratio"]].mean()
    cluster_summary["risk_score"] = (
        -cluster_summary["ROA"] * 2
        - cluster_summary["capital_adequacy_ratio"]
        + cluster_summary["npl_ratio"]
    )
    rank = cluster_summary["risk_score"].rank(method="first").fillna(1).astype(int)
    label_map = {1: "Low Risk", 2: "Elevated Risk", 3: "High Risk", 4: "Critical Risk"}
    labels = {int(k): label_map[int(v)] for k, v in rank.items()}
    df["risk_cluster_label"] = df["cluster"].map(labels)

    centers_df = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_), columns=FEATURES
    )
    centers_df["cluster_label"] = [labels[i] for i in range(4)]
    centers_df.to_csv(os.path.join(MODEL_DIR, "cluster_centers.csv"), index=False)

    print("\n  K-Means clusters:")
    for label, count in df["risk_cluster_label"].value_counts().items():
        print(f"    {label}: {count:,} banks")

    return df


# ── Score All Banks ───────────────────────────────────────────────────────────

def score_institutions(df: pd.DataFrame, rf_pipeline) -> pd.DataFrame:
    X = df[FEATURES]
    df = df.copy()
    df["failure_probability"] = rf_pipeline.predict_proba(X)[:, 1]
    df["predicted_at_risk"]   = rf_pipeline.predict(X)
    df["risk_score_100"]      = (df["failure_probability"] * 100).round(1)
    df["flag_low_capital"]    = (pd.to_numeric(df["capital_adequacy_ratio"], errors="coerce") < 10).astype(int)
    df["flag_high_npl"]       = (pd.to_numeric(df["npl_ratio"], errors="coerce") > 5).astype(int)
    df["flag_negative_roa"]   = (pd.to_numeric(df["ROA"], errors="coerce") < 0).astype(int)
    df["flag_high_ldr"]       = (pd.to_numeric(df["loan_to_deposit_ratio"], errors="coerce") > 90).astype(int)
    df["total_flags"]         = df[["flag_low_capital", "flag_high_npl", "flag_negative_roa", "flag_high_ldr"]].sum(axis=1)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def run_ml():
    print("\n=== ML Risk Engine ===\n")

    print("[1/4] Loading data...")
    institutions, failures = load_data()

    print("\n[2/4] Preparing dataset...")
    df_model = prepare_supervised_dataset(institutions, failures)

    print("\n[3/4] Training supervised models...")
    results = train_supervised(df_model)

    print("\n[4/4] Clustering banks by risk profile...")
    df_clustered = train_clustering(df_model)

    # Score all banks
    rf_pipe = results["Random Forest"]["pipeline"]
    df_scored = score_institutions(df_clustered, rf_pipe)

    # Save to SQLite
    conn = sqlite3.connect(DB_PATH)
    df_scored.drop(columns=["pipeline"], errors="ignore").to_sql(
        "risk_scores", conn, if_exists="replace", index=False
    )
    conn.close()

    # Save CSVs
    data_dir = os.path.normpath(os.path.join(MODEL_DIR, ".."))
    df_scored.to_csv(os.path.join(data_dir, "risk_scores.csv"), index=False)
    results["feature_importance"].to_csv(os.path.join(MODEL_DIR, "feature_importance.csv"), index=False)

    print("\n=== ML Complete ===")
    print(f"  Scored {len(df_scored):,} banks")
    print(f"  Models saved to: {os.path.abspath(MODEL_DIR)}")
    return df_scored, results


if __name__ == "__main__":
    run_ml()