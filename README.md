# 🛡 PaySecure — UPI Fraud Detection using Statistical Inference

> **First student project to apply a full statistical inference pipeline — MLE, Hypothesis Testing, and Regularized Regression — on primary UPI transaction survey data, deployed as a live fraud detection engine.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-paysecure.onrender.com-4F46E5?style=for-the-badge)](https://paysecure.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?style=for-the-badge)](https://render.com)
[![AUC](https://img.shields.io/badge/AUC--ROC-0.90-success?style=for-the-badge)]()

---

## What is PaySecure?

UPI fraud in India is rising rapidly. Most detection systems rely on large corporate datasets and black-box models. PaySecure takes a different approach.

> We collected **real data** from **202 people**, applied **rigorous statistical inference**, and built a **live fraud predictor** — all from scratch.

Instead of downloading a dataset, we ran a Google Forms survey capturing real UPI behaviour — amounts, time of day, recipient type, OTP requests, and more. We then used classical statistics to understand the data before training any model.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | Primary survey — Google Forms |
| Total Responses | 202 real respondents |
| Fraud Cases | 49 (24.3%) |
| Legit Cases | 153 (75.7%) |
| Features Used | 12 |
| Target | `is_fraud` (1 = Fraud, 0 = Legit) |

**No Kaggle. No external datasets. 100% original data.**

---

## 🧪 Statistical Pipeline

```
Survey Data → Preprocessing → Inference → Modelling → API → Live App
```

### Hypothesis Tests

| Test | H₀ | p-value | Result |
|------|----|---------|--------|
| Z-Test | Night fraud rate = Day fraud rate | 0.1637 | Not Significant |
| Chi-Square | Fraud ⊥ Recipient Type | 0.3828 | Not Significant |
| T-Test | Amount same for fraud & legit | 0.5285 | Not Significant |
| MLE | Age ~ Normal(μ, σ) | — | μ estimated |
| 95% CI | True fraud rate | — | [18.4% – 30.2%] |

### Models

| Model | Technique | Result |
|-------|-----------|--------|
| Logistic Regression | MLE-based binary classification | **AUC = 0.90** |
| Ridge Regression | L2 regularization (α = 1.0) | All 12 features retained |
| Lasso Regression | L1 regularization + feature selection | Weak features zeroed out |

---

## 🗂 Project Structure

```
paysecure/
├── data/
│   └── paysecure_dataset.xlsx   ← 202 real survey responses
├── static/
│   └── index.html               ← full frontend (HTML + CSS + JS)
├── preprocess.py                ← data cleaning & encoding
├── model.py                     ← stats tests + model training
├── api.py                       ← FastAPI (5 endpoints)
├── requirements.txt
└── README.md
```

---

## 📱 App Pages

| Page | What It Does |
|------|-------------|
| 🏠 Home | Live stats, fraud distribution chart, fraud by time chart |
| 🎯 Fraud Predictor | 11-field form → real-time fraud probability score |
| 📊 Dashboard | Age histogram (MLE), ROC curve, Ridge coefficients |
| ⚗ Inference Lab | All 5 statistical tests with p-values + plain English |
| 📈 Model Report | AUC, Ridge α, Lasso feature table |

---

## ⚙️ Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.11 |
| Statistics & ML | scipy · scikit-learn · numpy |
| Data | pandas · openpyxl |
| Backend | FastAPI · uvicorn |
| Frontend | HTML · CSS · JavaScript · Chart.js |
| Deployment | Render · GitHub |

---

## 🚀 Run Locally

```bash
git clone https://github.com/diyam-03/paysecure.git
cd paysecure

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
python preprocess.py
python model.py

uvicorn api:app --reload
# Open http://localhost:8000
```

---

## 👩‍💻 Author

**Diya Mehta** · NMIMS MPSTME · Semester IV  
**Rugved Tatkare** · NMIMS MPSTME · Semester IV

Subject: Statistical Structures in Data and 
