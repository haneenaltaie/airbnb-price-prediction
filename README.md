# Airbnb Price Prediction — Complete ML Pipeline
**AIGC 5003 — Machine Learning in Cloud Computing**  
**Author: Haneen Altaie | AI Integration and Governance, Humber College**

---

## Project Overview
This project builds a complete end-to-end machine learning pipeline to predict 
Airbnb nightly listing prices using Amazon SageMaker. The pipeline covers data 
loading, feature engineering, cloud-based training, hyperparameter tuning, 
real-time endpoint deployment, and resource cleanup.

---

## Final Results

| Metric | Value |
|--------|-------|
| MAE | $54.98 |
| RMSE | $74.23 |
| R² | 0.2109 |
| Features | 529 engineered features |
| Dataset size | 953 listings |
| Outliers removed | 48 listings above $441 |
| Solo Training RMSE | 76.98 |
| HP Tuning Best RMSE | 72.75 |
| MAE improvement | 26% better than baseline |
| RMSE improvement | 41% better than baseline |

---

## Pipeline Steps

| Step | Description | Key Output |
|------|-------------|------------|
| 1 | Data Loading & Analysis | EDA plots, basic statistics |
| 2 | Feature Engineering | 529 features including location |
| 3 | SageMaker XGBoost Training Job | validation:rmse = 76.98 |
| 4 | Hyperparameter Tuning (10 jobs) | Best validation:rmse = 72.75 |
| 5 | Model Deployment (ml.m5.large) | Real-time endpoint |
| 6 | Inferencing & Validation | MAE=$54.98, RMSE=$74.23, R²=0.2109 |
| 7 | Resource Cleanup | Endpoint deleted |

---

## Feature Engineering
Features extracted from raw listing data:

| Feature | Description |
|---------|-------------|
| `beds` | Number of beds extracted from text |
| `rating` | Numerical score from review column |
| `num_reviews` | Review count from parentheses |
| `discount_pct` | Discount between price and offer price |
| `detail_length` | Character count of listing description |
| `is_luxury` | Keyword flag from description |
| `has_hot_tub` | Keyword flag from description |
| `has_pool` | Keyword flag from description |
| `has_pets` | Keyword flag from description |
| `is_treehouse` | Keyword flag from description |
| `property_type` | Extracted from title, one-hot encoded |
| `state` | Extracted from title (e.g. Washington, New York) |
| `city` | Extracted from title (e.g. Skykomish, Hancock) |

Adding `state` and `city` location features increased feature count 
from 42 to 529 and improved R² from 0.1852 to 0.2109 — a 14% improvement.

---

## Models Used

- **SageMaker Built-in XGBoost** — used for training job and hyperparameter tuning
- **Random Forest Regressor (scikit-learn)** — used for the deployed endpoint

---

## AWS SageMaker Deployment

Deployment workflow:
1. Upload data splits to Amazon S3
2. Train XGBoost model on separate `ml.m5.xlarge` instance
3. Run 10 hyperparameter tuning jobs automatically
4. Train Random Forest v3 on 529 engineered features
5. Package model as `model.tar.gz` and upload to S3
6. Deploy real-time endpoint on `ml.m5.large` using scikit-learn 1.4-2
7. Validate endpoint — results match local model exactly
8. Delete endpoint immediately to avoid per-hour charges

---

## Deployment Challenges Solved

| Issue | Fix |
|-------|-----|
| sklearn version mismatch | Retrained with scikit-learn 1.4.2 |
| ml.t2.medium memory error | Switched to ml.m5.large |
| KeyError on price(in_dollar) | Used positional indexing |
| %%writefile in wrong cell | Moved to separate cell |

---

## Tech Stack

- Python, Pandas, NumPy, Scikit-learn, XGBoost
- Matplotlib, Seaborn
- Amazon SageMaker, Amazon S3
- Joblib, Tarfile
- Region: ca-central-1

---

## Project Files

| File | Description |
|------|-------------|
| `airbnb-sagemaker.ipynb` | Main notebook — complete pipeline |
| `Airbnb_Price_Pipeline.pdf` | Exported notebook with all outputs |
| `inference.py` | Custom inference script for endpoint |
| `airbnb_final_report.pdf` | Final written report |
| `requirements.txt` | Required Python libraries |

---

## How to Run

1. Open `airbnb-sagemaker.ipynb` in SageMaker JupyterLab
2. Install required libraries:
```bash
pip install -r requirements.txt
```
3. Run all cells in order (Kernel → Restart & Run All)
4. Review results in the final summary cell

---

## Key Learning Outcomes

- End-to-end ML pipeline development on AWS
- Feature engineering from raw text data
- Cloud-based training on separate compute instances
- Automated hyperparameter tuning with SageMaker
- Real-time model deployment and endpoint validation
- Debugging real cloud deployment issues
- Cost management (endpoint cleanup)

---

## Conclusion

This project demonstrates a complete machine learning workflow:

**data cleaning → feature engineering → cloud training → 
hyperparameter tuning → deployment → validation → cleanup**

Built to gain hands-on experience with real-world cloud ML deployment 
on Amazon SageMaker, while achieving a 26% improvement in MAE over baseline.
