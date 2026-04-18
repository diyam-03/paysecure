from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle, os
from preprocess import load_and_clean
from model import run_all

app = FastAPI()

# ── Numpy serialization fix ───────────────────────────────────────────────────
def np_clean(obj):
    if isinstance(obj, dict):
        return {k: np_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [np_clean(i) for i in obj]
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.bool_): return bool(obj)
    return obj

# ── Bootstrap data & model on startup ────────────────────────────────────────
if not os.path.exists("data/cleaned.csv"):
    load_and_clean()

if not os.path.exists("models/logistic_model.pkl"):
    run_all()

with open("models/logistic_model.pkl", "rb") as f:
    pkg = pickle.load(f)

df = pd.read_csv("data/cleaned.csv")
results = run_all()

# ── Serve frontend ────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")

# ── API endpoints ─────────────────────────────────────────────────────────────
@app.get("/api/stats")
def get_stats():
    fraud_count = int(df["is_fraud"].sum())
    total = len(df)
    ci = results["fraud_ci"]
    return np_clean({
        "total": total,
        "fraud": fraud_count,
        "legit": total - fraud_count,
        "fraud_rate": round(float(df["is_fraud"].mean()) * 100, 1),
        "ci_lower": round(float(ci["lower"]) * 100, 1),
        "ci_upper": round(float(ci["upper"]) * 100, 1),
        "auc": results["logistic"]["auc"],
        "mle_mu": results["mle_age"]["mu"],
        "mle_std": results["mle_age"]["std"],
    })

@app.get("/api/charts")
def get_charts():
    time_fraud = df.groupby("time_of_txn")["is_fraud"].mean().round(3).to_dict()
    recip_fraud = df.groupby("recipient_type")["is_fraud"].mean().round(3).to_dict()
    app_fraud = df.groupby("upi_app")["is_fraud"].mean().round(3).to_dict()
    ages = df["age"].dropna().tolist()
    return np_clean({
        "time_fraud": time_fraud,
        "recip_fraud": recip_fraud,
        "app_fraud": app_fraud,
        "ages": ages,
        "roc_fpr": results["logistic"]["fpr"],
        "roc_tpr": results["logistic"]["tpr"],
        "ridge_coef": results["ridge"]["coef"],
        "lasso_coef": results["lasso"],
    })

@app.get("/api/tests")
def get_tests():
    z = results["ztest_night"]
    chi = results["chi_square"]
    t = results["ttest_amount"]
    mle = results["mle_age"]
    ci = results["fraud_ci"]
    return np_clean({
        "ztest": {
            "stat": z["z_stat"], "p": z["p_value"],
            "night_rate": round(float(z["night_fraud_rate"]) * 100, 1),
            "day_rate": round(float(z["day_fraud_rate"]) * 100, 1),
            "significant": bool(z["p_value"] < 0.05)
        },
        "chi": {
            "stat": chi["chi2"], "p": chi["p_value"],
            "dof": chi["dof"],
            "significant": bool(chi["p_value"] < 0.05)
        },
        "ttest": {
            "stat": t["t_stat"], "p": t["p_value"],
            "fraud_mean": t["fraud_mean"],
            "legit_mean": t["nonfraud_mean"],
            "significant": bool(t["p_value"] < 0.05)
        },
        "mle": {"mu": mle["mu"], "std": mle["std"]},
        "ci": {
            "lower": round(float(ci["lower"]) * 100, 1),
            "upper": round(float(ci["upper"]) * 100, 1),
            "p": round(float(ci["p"]) * 100, 1)
        }
    })

class TxnInput(BaseModel):
    age: int
    new_recipient: int
    diff_location: int
    multiple_txns: int
    suspicious_link: int
    asked_otp: int
    amount_encoded: int
    freq_encoded: int
    gender_enc: int
    upi_app_enc: int
    time_of_txn_enc: int
    recipient_type_enc: int

@app.post("/api/predict")
def predict(txn: TxnInput):
    feature_cols = ["age","new_recipient","diff_location","multiple_txns",
                    "suspicious_link","asked_otp","amount_encoded","freq_encoded",
                    "gender_enc","upi_app_enc","time_of_txn_enc","recipient_type_enc"]
    X = pd.DataFrame([txn.dict()])[feature_cols]
    X_scaled = pkg["scaler"].transform(X)
    prob = float(pkg["model"].predict_proba(X_scaled)[0][1])

    reasons = []
    if txn.asked_otp:       reasons.append("OTP/PIN was requested")
    if txn.suspicious_link: reasons.append("Suspicious payment link received")
    if txn.new_recipient:   reasons.append("Transaction to a new recipient")
    if txn.diff_location:   reasons.append("Different location than usual")
    if txn.multiple_txns:   reasons.append("Multiple transactions in short time")

    return {
        "probability": round(prob * 100, 1),
        "risk": "High Risk" if prob > 0.5 else "Low Risk",
        "high": bool(prob > 0.5),
        "reasons": reasons[:3]
    }