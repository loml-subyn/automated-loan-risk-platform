"""
================================================================================
  LOAN APPROVAL PREDICTOR — Dream Housing Finance
  ------------------------------------------------
  Binary classification pipeline comparing Logistic Regression vs Decision Tree.

  Exports FOUR production artifacts:
    1. loan_model_lr.pkl   — Trained Logistic Regression model
    2. loan_model_dt.pkl   — Trained Decision Tree Classifier
    3. scaler.pkl          — Fitted StandardScaler (continuous features only)
    4. model_columns.pkl   — Ordered list of feature column names post-encoding

  Author : Auto-generated production script
  Python : 3.8+
  Deps   : numpy, pandas, scikit-learn, joblib
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
)
import joblib
import os

# ──────────────────────────────────────────────────────────────────────────────
# PART 1: DATA SIMULATION & INITIALIZATION
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  PART 1 — DATA SIMULATION & INITIALIZATION")
print("=" * 70)

np.random.seed(42)
N = 500

# --- Categorical features ---
gender = np.random.choice(["Male", "Female"], size=N, p=[0.65, 0.35])
married = np.random.choice(["Yes", "No"], size=N, p=[0.60, 0.40])
education = np.random.choice(["Graduate", "Not Graduate"], size=N, p=[0.70, 0.30])
property_area = np.random.choice(
    ["Urban", "Semiurban", "Rural"], size=N, p=[0.35, 0.35, 0.30]
)

# --- Continuous numerical features ---
applicant_income = np.random.randint(2500, 15001, size=N)
coapplicant_income = np.round(np.random.uniform(0, 5000, size=N), 2)

# --- LoanAmount with ~5 % deliberate NaN values ---
loan_amount = np.random.randint(50, 301, size=N).astype(float)
nan_loan_idx = np.random.choice(N, size=int(0.05 * N), replace=False)
loan_amount[nan_loan_idx] = np.nan

# --- Credit_History with ~5 % deliberate NaN values ---
credit_history = np.random.choice([1.0, 0.0], size=N, p=[0.70, 0.30])
nan_credit_idx = np.random.choice(N, size=int(0.05 * N), replace=False)
credit_history[nan_credit_idx] = np.nan

# --- Target: Loan_Status heavily correlated with Credit_History ---
#     If Credit_History == 1.0 → 85 % chance of approval (Y)
#     If Credit_History == 0.0 → 20 % chance of approval (Y)
#     If Credit_History is NaN → 50 % chance (coin flip)
loan_status = []
for ch in credit_history:
    if np.isnan(ch):
        loan_status.append(np.random.choice(["Y", "N"], p=[0.50, 0.50]))
    elif ch == 1.0:
        loan_status.append(np.random.choice(["Y", "N"], p=[0.85, 0.15]))
    else:
        loan_status.append(np.random.choice(["Y", "N"], p=[0.20, 0.80]))
loan_status = np.array(loan_status)

# --- Assemble the DataFrame ---
df = pd.DataFrame(
    {
        "Gender": gender,
        "Married": married,
        "Education": education,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Credit_History": credit_history,
        "Property_Area": property_area,
        "Loan_Status": loan_status,
    }
)

print(f"\nDataset shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nTarget distribution:\n{df['Loan_Status'].value_counts()}")

# ──────────────────────────────────────────────────────────────────────────────
# PART 2: EXPLORATORY DATA ANALYSIS & PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  PART 2 — EXPLORATORY DATA ANALYSIS & PREPROCESSING")
print("=" * 70)

# 1. Missing value count per column
print("\n--- Missing Values Per Column ---")
missing = df.isnull().sum()
print(missing)
print(f"\nTotal missing cells: {missing.sum()}")

# 2. Impute missing numerical data (LoanAmount) with the median
loan_median = df["LoanAmount"].median()
df["LoanAmount"] = df["LoanAmount"].fillna(loan_median)
print(f"\n[IMPUTED] LoanAmount NaNs filled with median = {loan_median}")

# 3. Impute missing categorical data (Credit_History) with the mode
credit_mode = df["Credit_History"].mode()[0]
df["Credit_History"] = df["Credit_History"].fillna(credit_mode)
print(f"[IMPUTED] Credit_History NaNs filled with mode  = {credit_mode}")

# Verify no remaining nulls
assert df.isnull().sum().sum() == 0, "Imputation incomplete — nulls remain!"
print("\n✓ All missing values handled. Remaining NaNs: 0")

# 4. Separate features (X) and target (y); encode target to binary
y = df["Loan_Status"].map({"Y": 1, "N": 0})
X = df.drop(columns=["Loan_Status"])

print(f"\nFeature matrix shape : {X.shape}")
print(f"Target vector shape  : {y.shape}")
print(f"Target class balance : {dict(y.value_counts())}")

# 5. One-Hot Encoding for categorical features
X = pd.get_dummies(X, drop_first=True)
print(f"\nColumns after One-Hot Encoding ({X.shape[1]} features):")
print(list(X.columns))

# 6. Train / Test split (80/20, stratified, reproducible)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test  set: {X_test.shape[0]} samples")

# 7. Feature scaling (StandardScaler) — fit on train, transform both
#    Only scale continuous numerical columns; leave binary dummies as-is.
continuous_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]

scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])

print(f"\n✓ StandardScaler fitted on training set and applied to both sets.")
print(f"  Scaled columns: {continuous_cols}")

# ──────────────────────────────────────────────────────────────────────────────
# PART 3: LOGISTIC REGRESSION PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  PART 3 — LOGISTIC REGRESSION PIPELINE")
print("=" * 70)

lr_model = LogisticRegression(random_state=42, max_iter=1000, solver="lbfgs")
lr_model.fit(X_train_scaled, y_train)
lr_preds = lr_model.predict(X_test_scaled)

print(f"\n✓ Logistic Regression trained on {X_train_scaled.shape[0]} samples.")
print(f"  Predictions generated for {X_test_scaled.shape[0]} test samples.")

# ──────────────────────────────────────────────────────────────────────────────
# PART 4: DECISION TREE PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  PART 4 — DECISION TREE PIPELINE")
print("=" * 70)

dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)  # Unscaled data — trees are scale-invariant
dt_preds = dt_model.predict(X_test)

print(f"\n✓ Decision Tree (max_depth=3) trained on {X_train.shape[0]} samples.")
print(f"  Predictions generated for {X_test.shape[0]} test samples.")

# ──────────────────────────────────────────────────────────────────────────────
# PART 5: COMPREHENSIVE MODEL EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  PART 5 — COMPREHENSIVE MODEL EVALUATION")
print("=" * 70)


def evaluate_model(name: str, y_true, y_pred):
    """Print accuracy, confusion matrix, and classification report."""
    print(f"\n{'─' * 50}")
    print(f"  {name}")
    print(f"{'─' * 50}")

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["Rejected (0)", "Approved (1)"])

    print(f"\n  Accuracy: {acc:.4f}  ({acc * 100:.2f}%)")

    print(f"\n  Confusion Matrix:")
    print(f"                  Predicted: 0    Predicted: 1")
    print(f"  Actual: 0       {cm[0][0]:>10}    {cm[0][1]:>10}")
    print(f"  Actual: 1       {cm[1][0]:>10}    {cm[1][1]:>10}")

    print(f"\n  Classification Report:")
    print(report)

    return acc, cm


# --- Evaluate Logistic Regression ---
lr_acc, lr_cm = evaluate_model("LOGISTIC REGRESSION", y_test, lr_preds)

# --- Evaluate Decision Tree ---
dt_acc, dt_cm = evaluate_model("DECISION TREE (max_depth=3)", y_test, dt_preds)

# ──────────────────────────────────────────────────────────────────────────────
# FINAL COMPARISON SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  FINAL COMPARATIVE SUMMARY")
print("=" * 70)

lr_f1 = f1_score(y_test, lr_preds, pos_label=1)
dt_f1 = f1_score(y_test, dt_preds, pos_label=1)

lr_fp = lr_cm[0][1]  # False Positives for LR
dt_fp = dt_cm[0][1]  # False Positives for DT

print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │                    METRIC COMPARISON TABLE                   │
  ├──────────────────────┬────────────────────┬─────────────────-┤
  │  Metric              │ Logistic Regr.     │ Decision Tree    │
  ├──────────────────────┼────────────────────┼──────────────────┤
  │  Accuracy            │ {lr_acc:>17.4f}  │ {dt_acc:>15.4f}  │
  │  F1 (Approved=1)     │ {lr_f1:>17.4f}  │ {dt_f1:>15.4f}  │
  │  False Positives     │ {lr_fp:>17}  │ {dt_fp:>15}  │
  └──────────────────────┴────────────────────┴──────────────────┘
""")

# --- Highlight the winner ---
if lr_f1 > dt_f1:
    f1_winner = "Logistic Regression"
    f1_loser = "Decision Tree"
elif dt_f1 > lr_f1:
    f1_winner = "Decision Tree"
    f1_loser = "Logistic Regression"
else:
    f1_winner = None

if f1_winner:
    print(f"  ▸ F1-Score (Approved class):  {f1_winner} wins "
          f"({max(lr_f1, dt_f1):.4f} vs {min(lr_f1, dt_f1):.4f}).")
else:
    print(f"  ▸ F1-Score (Approved class):  Both models are tied at {lr_f1:.4f}.")

if lr_fp < dt_fp:
    fp_winner = "Logistic Regression"
elif dt_fp < lr_fp:
    fp_winner = "Decision Tree"
else:
    fp_winner = None

if fp_winner:
    print(f"  ▸ False Positives:  {fp_winner} is better at minimizing False Positives "
          f"({min(lr_fp, dt_fp)} vs {max(lr_fp, dt_fp)}).")
    print(f"    → Fewer False Positives means fewer risky loans approved incorrectly.")
else:
    print(f"  ▸ False Positives:  Both models produced the same number ({lr_fp}).")

print(f"""
  ════════════════════════════════════════════════════════════════
  CONCLUSION:
  In a loan-approval setting, minimizing False Positives (approving
  someone who should be rejected) is critical for financial risk.
  Based on the metrics above, {'**' + fp_winner + '**' if fp_winner else 'both models'} 
  {'is' if fp_winner else 'are'} the safer choice for deployment.
  ════════════════════════════════════════════════════════════════
""")

print("✓ Pipeline complete. All models trained, evaluated, and compared.")

# ──────────────────────────────────────────────────────────────────────────────
# EXPORT: Save FOUR production artifacts for the Streamlit app
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  EXPORT — SAVING PRODUCTION ARTIFACTS")
print("=" * 70)

export_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Trained Logistic Regression model
lr_path = os.path.join(export_dir, "loan_model_lr.pkl")
joblib.dump(lr_model, lr_path)
print(f"\n  [1/4] ✓ Logistic Regression model  → '{lr_path}'")

# 2. Trained Decision Tree model
dt_path = os.path.join(export_dir, "loan_model_dt.pkl")
joblib.dump(dt_model, dt_path)
print(f"  [2/4] ✓ Decision Tree model        → '{dt_path}'")

# 3. Fitted StandardScaler
scaler_path = os.path.join(export_dir, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"  [3/4] ✓ StandardScaler (fitted)     → '{scaler_path}'")

# 4. Feature column names (exact order post-encoding)
model_columns = list(X.columns)
columns_path = os.path.join(export_dir, "model_columns.pkl")
joblib.dump(model_columns, columns_path)
print(f"  [4/4] ✓ Model feature columns       → '{columns_path}'")

print(f"\n  Exported feature columns: {model_columns}")
print(f"\n{'═' * 70}")
print(f"  ALL 4 ARTIFACTS SUCCESSFULLY ARCHIVED FOR DEPLOYMENT.")
print(f"  Ready for Streamlit app consumption.")
print(f"{'═' * 70}")
