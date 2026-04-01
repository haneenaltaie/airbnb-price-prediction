# Airbnb Nightly Price Prediction — Complete ML Pipeline

> **AIGC 5003 — Machine Learning in Cloud Computing**  
> Humber College | Haneen Al-Taie | N01073800 | April 2026

---

## Project Overview

This project implements a complete end-to-end machine learning pipeline on **Amazon SageMaker** to predict Airbnb nightly listing prices. Starting from raw listing data, the pipeline covers data loading, feature engineering, model training, hyperparameter tuning, real-time endpoint deployment, and inferencing.

**Problem Statement:**
> *"Prediction of Airbnb nightly listing prices using property features, host attributes, seasonal demand, and location data."*

---

## Repository Structure

```
├── Airbnb_Price_Pipeline.ipynb   ✅
├── Airbnb_Price_Pipeline.pdf     ✅
├── Airbnb_Report_.pdf            ✅
├── airnb.xls                     ✅ (was airbnb.csv before)
├── inference.py                  ✅
├── README.md                     ✅
```

---

## Pipeline Overview

| Step | Description | Tool |
|------|-------------|------|
| 1 | Data Loading & Analysis | pandas, seaborn |
| 2 | Cleaning & Feature Engineering (537 features) | pandas, regex |
| 3 | BM1 — XGBoost baseline training | SageMaker built-in XGBoost |
| 4 | BM2 — Bayesian hyperparameter tuning | SageMaker HyperparameterTuner |
| 5 | BM3 — Random Forest deployment | SageMaker SKLearnModel |
| 6 | Inferencing & Evaluation | SageMaker endpoint |
| 7 | Resource Cleanup | boto3 |
| 8 | Reflection & Data Insights | matplotlib, numpy |

---

## Dataset

| Property | Value |
|----------|-------|
| Source | Raw Airbnb listings |
| Raw rows | 953 |
| Raw columns | 7 (Title, Detail, Date, Price, Offer Price, Review & Rating, Beds) |
| After cleaning | 905 rows |
| Features engineered | 537 |
| Outlier cutoff | 95th percentile ($441) |
| Target variable | `price(in_dollar)` |

### Features Engineered from Raw Data

| Feature | Description |
|---------|-------------|
| `beds` | Number of beds extracted from text |
| `rating` | Numerical score from review column |
| `num_reviews` | Review count from parentheses |
| `is_luxury` | Keyword flag from listing description |
| `has_hot_tub` | Keyword flag from listing description |
| `has_pool` | Keyword flag from listing description |
| `has_pets` | Keyword flag from listing description |
| `property_type` | Extracted from title, one-hot encoded |
| `detail_length` | Character count of listing description |
| `discount_pct` | Percentage discount between price and offer price |
| `state` | Extracted from title (e.g. Washington, New York) |
| `city` | Extracted from title (e.g. Skykomish, Hancock) |
| `season` | Season from listing date (Summer, Winter, Spring, Fall) |
| `quarter` | Quarter from listing date (Q1, Q2, Q3, Q4) |

---

## Benchmark Results

| Benchmark | Dataset | Features | Metric | Result | vs Baseline |
|-----------|---------|----------|--------|--------|-------------|
| BM1 — XGBoost default HP | sagemaker_train.csv | 537 engineered | validation:rmse | **71.84** | baseline |
| BM2 — XGBoost tuned HP | sagemaker_train.csv | 537 engineered | validation:rmse | **66.77** | ↓ -5.07 |
| BM3 — Random Forest endpoint | 537 engineered | RMSE / MAE / R² | **71.62 / $52.64 / 0.2655** | deployed |

### Best Hyperparameters (BM2)

| Parameter | Value |
|-----------|-------|
| eta | 0.1778 |
| max_depth | 4 |
| num_round | 108 |
| subsample | 0.8998 |
| objective | reg:squarederror |

### Deployed Endpoint Results (BM3)

| Metric | Value |
|--------|-------|
| MAE | $52.64 |
| RMSE | $71.62 |
| R² | 0.2655 |

---

## Key Findings

1. **Season** — Spring listings average $187 (most expensive). Summer has the most listings (583) but lowest average price ($129) — high supply drives prices down.
2. **Quarter** — Q3 (Jul–Sep) is the most expensive quarter at $154.90/night.
3. **Location** — North Carolina ($235) and California ($227) command the highest prices. Location is the strongest single price driver.
4. **Amenities** — A hot tub adds the most value (+$76), more than a luxury label (+$35). Pool and pet-friendly listings showed slight negative correlation with price.
5. **Property Type** — Dome ($235) and Villa ($202) are the most premium types. Hotels ($67) and Rooms ($98) are the most affordable.
6. **Feature Engineering vs Tuning** — Feature engineering improved RMSE by 5.07 points. This proved that investing in meaningful data preparation is more valuable than hyperparameter tuning alone.

---

## Infrastructure

| Setting | Value |
|---------|-------|
| Region | ca-central-1 (Canada Central) |
| Training instance | ml.m5.xlarge |
| Hosting instance | ml.m5.large |
| Framework | scikit-learn 1.4-2 |
| XGBoost version | 1.7-1 |
| S3 bucket | sagemaker-ca-central-1-263245924849 |

---

## How to Run

> **Requirements:** AWS account with SageMaker access, JupyterLab environment on SageMaker

1. Upload `airbnb.csv` and `final_airbnb_ml.csv` to your SageMaker JupyterLab instance
2. Upload `Airbnb_Price_Pipeline.ipynb` and `inference.py`
3. Open the notebook and run all cells top to bottom
4. All AWS resources (endpoints) are deleted automatically in Step 7

> ⚠️ Make sure to run Step 7 (Resource Cleanup) to avoid unnecessary AWS charges

---

## Libraries Used

```python
pandas, numpy, matplotlib, seaborn      # Data analysis & visualization
scikit-learn                            # Random Forest, metrics, train_test_split
boto3                                   # AWS SDK
sagemaker                               # SageMaker Python SDK
joblib, tarfile                         # Model packaging
re, os                                  # Feature engineering utilities
```

---

## Author

**Haneen Al-Taie**  
Student ID: N01073800  
Program: AI Integration and Governance  
Humber College | April 2026
