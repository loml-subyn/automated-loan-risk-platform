# 🏛️ Institutional Credit Risk Assessment Platform
> Automated Credit Scoring, Risk Profiling, and Decisioning Engine.

---

## 📋 Project Overview
This repository contains an end-to-end, production-grade machine learning framework designed to replicate an institutional-grade automated underwriting platform used by modern financial risk departments. 

The system trains and evaluates two distinct classification architectures—Logistic Regression and a Decision Tree Classifier—to predict loan approval probabilities based on historical applicant risk profiles. To bridge the gap between core data science and real-world software engineering, the best-performing model is deployed as an interactive, corporate-themed dashboard interface.

---

## ✨ Key Features
* **Dual-Model Predictive Pipeline:** Compares a linear model (Logistic Regression) against a non-linear branch-structured model (Decision Tree) with explicit optimization for minimizing expensive False Positives (high-risk defaults).
* **Robust Data Guardrails:** Advanced preprocessing logic featuring data-leakage prevention (fit-transform scaling restricted to the training split) and automated category imputation.
* **Flawless Array Alignment:** Implements deterministic inference mapping via schema tracking (`model_columns.pkl`) to guarantee input feature vectors match the training matrix layout precisely.
* **FinTech Enterprise UI:** A custom-styled, wide-layout dark theme dashboard built in Streamlit, mirroring a modern banking software terminal.
* **Functional CIBIL Report Processing:** Integrates an automated document processing layer that simulates reading and extracting 3-digit credit scores directly from uploaded PDFs.
* **Explainable AI (XAI):** Features a live feature-attribution matrix mapping model coefficients to individual risk metrics, giving underwriters transparent visibility into the exact variables driving an approval or rejection.

---

## 🛠️ Tech Stack & Dependencies
* **Core Language:** Python 3.8+
* **Machine Learning Library:** scikit-learn
* **Data Manipulation:** pandas, numpy
* **Web Frontend Dashboard:** streamlit
* **Model Serialization:** joblib
* **Parsing Ecosystem:** pdfplumber / pypdf

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/credit-underwriting-engine.git](https://github.com/yourusername/credit-underwriting-engine.git)
cd credit-underwriting-engine
