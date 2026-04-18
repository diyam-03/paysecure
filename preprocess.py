import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def load_and_clean():
    df = pd.read_excel("data/paysecure_dataset.xlsx")

    # Drop unnamed index column
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c])

    # Rename columns to short names
    df.columns = [
        "age", "gender", "city", "upi_app", "amount_range",
        "time_of_txn", "recipient_type", "new_recipient",
        "diff_location", "multiple_txns", "suspicious_link",
        "asked_otp", "txn_frequency", "is_fraud"
    ]

    # Clean whitespace in all string columns
    str_cols = df.select_dtypes("object").columns
    df[str_cols] = df[str_cols].apply(lambda x: x.str.strip())

    # Standardize city casing
    df["city"] = df["city"].str.title()

    # Drop rows with missing target
    df = df.dropna(subset=["is_fraud"])

    # Binary encode Yes/No columns
    binary_cols = ["new_recipient", "diff_location", "multiple_txns",
                   "suspicious_link", "asked_otp"]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # Encode target
    df["is_fraud"] = df["is_fraud"].map({"Yes": 1, "No": 0})

    # Ordinal encode amount range
    amount_order = {
    "₹0–500": 1,       "₹0 – ₹500": 1,
    "₹500–2000": 2,    "₹500 – ₹2000": 2,
    "₹2000–10000": 3,  "₹2000 – ₹10,000": 3,
    "₹10000+": 4,      "₹10,000+": 4
}
    df["amount_encoded"] = df["amount_range"].map(amount_order)

    # Ordinal encode frequency
    freq_order = {
        "Low": 1,    "Low (1–2 transactions)": 1,
        "Medium": 2, "Medium (3–5 transactions)": 2,
        "High": 3,   "High (6+ transactions)": 3
    }
    df["freq_encoded"] = df["txn_frequency"].map(freq_order)
    df["freq_encoded"] = df["txn_frequency"].map(freq_order)

    # Label encode remaining categoricals
    le = LabelEncoder()
    for col in ["gender", "city", "upi_app", "time_of_txn", "recipient_type"]:
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/cleaned.csv", index=False)
    print(f"Saved cleaned.csv — {len(df)} rows, {df['is_fraud'].sum()} fraud cases")
    return df

if __name__ == "__main__":
    load_and_clean()