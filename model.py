import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle, os

def get_features_target(df):
    feature_cols = [
        "age", "new_recipient", "diff_location", "multiple_txns",
        "suspicious_link", "asked_otp", "amount_encoded", "freq_encoded",
        "gender_enc", "upi_app_enc", "time_of_txn_enc", "recipient_type_enc"
    ]
    df = df.dropna(subset=feature_cols + ["is_fraud"])
    X = df[feature_cols]
    y = df["is_fraud"]
    return X, y, feature_cols

# ── 1. MLE: Fit normal distribution to age ───────────────────────────────────
def mle_age(df):
    ages = df["age"].dropna()
    mu, std = stats.norm.fit(ages)
    return {"mu": round(mu, 2), "std": round(std, 2)}

# ── 2. Confidence Interval for fraud rate ────────────────────────────────────
def fraud_rate_ci(df, confidence=0.95):
    n = len(df)
    p = df["is_fraud"].mean()
    z = stats.norm.ppf((1 + confidence) / 2)
    margin = z * np.sqrt(p * (1 - p) / n)
    return {"p": round(p, 4), "lower": round(p - margin, 4), "upper": round(p + margin, 4), "n": n}

# ── 3. Z-Test: fraud rate at night vs day ────────────────────────────────────
def ztest_night_fraud(df):
    night_mask = df["time_of_txn"].str.contains("Night", na=False)
    night = df[night_mask]["is_fraud"]
    day   = df[~night_mask]["is_fraud"]
    z, p  = stats.ttest_ind(night, day)
    return {"z_stat": round(z, 4), "p_value": round(p, 4),
            "night_fraud_rate": round(night.mean(), 4),
            "day_fraud_rate": round(day.mean(), 4)}

# ── 4. Chi-Square: fraud vs recipient type ───────────────────────────────────
def chi_square_recipient(df):
    ct = pd.crosstab(df["recipient_type"], df["is_fraud"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    return {"chi2": round(chi2, 4), "p_value": round(p, 4), "dof": dof, "crosstab": ct}

# ── 5. T-Test: amount encoded fraud vs non-fraud ─────────────────────────────
def ttest_amount(df):
    fraud    = df[df["is_fraud"] == 1]["amount_encoded"].dropna()
    no_fraud = df[df["is_fraud"] == 0]["amount_encoded"].dropna()
    t, p = stats.ttest_ind(fraud, no_fraud)
    return {"t_stat": round(t, 4), "p_value": round(p, 4),
            "fraud_mean": round(fraud.mean(), 4),
            "nonfraud_mean": round(no_fraud.mean(), 4)}

# ── 6. Logistic Regression ───────────────────────────────────────────────────
def train_logistic(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_tr, y_train)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return model, scaler, {
        "auc": round(auc, 4),
        "report": report,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "confusion_matrix": cm.tolist()
    }

# ── 7. Ridge Classifier ──────────────────────────────────────────────────────
def train_ridge(X_train, X_test, y_train, y_test, scaler):
    X_tr = scaler.transform(X_train)
    X_te = scaler.transform(X_test)
    model = RidgeClassifier(alpha=1.0)
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    report = classification_report(y_test, y_pred, output_dict=True)
    coef = model.coef_.flatten()
    return {"report": report, "coef": dict(zip(X_train.columns, coef.round(4).tolist()))}

# ── 8. Lasso (feature selection via coefficients) ────────────────────────────
def run_lasso(X_train, y_train, scaler):
    X_tr = scaler.transform(X_train)
    model = Lasso(alpha=0.01, max_iter=5000)
    model.fit(X_tr, y_train)
    return dict(zip(X_train.columns, model.coef_.round(4).tolist()))

# ── MAIN ─────────────────────────────────────────────────────────────────────
def run_all():
    df = pd.read_csv("data/cleaned.csv")
    X, y, feat_cols = get_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    scaler.fit(X_train)

    results = {
        "mle_age":           mle_age(df),
        "fraud_ci":          fraud_rate_ci(df),
        "ztest_night":       ztest_night_fraud(df),
        "chi_square":        chi_square_recipient(df),
        "ttest_amount":      ttest_amount(df),
    }

    log_model, scaler, log_res = train_logistic(X_train, X_test, y_train, y_test)
    results["logistic"] = log_res
    results["ridge"]    = train_ridge(X_train, X_test, y_train, y_test, scaler)
    results["lasso"]    = run_lasso(X_train, y_train, scaler)
    results["features"] = feat_cols

    os.makedirs("models", exist_ok=True)
    with open("models/logistic_model.pkl", "wb") as f:
        pickle.dump({"model": log_model, "scaler": scaler, "features": feat_cols}, f)

    print("✅ Model trained | AUC:", results["logistic"]["auc"])
    return results

if __name__ == "__main__":
    from preprocess import load_and_clean
    load_and_clean()
    run_all()