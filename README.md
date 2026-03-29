# Airbnb Price Prediction Using Machine Learning

**Author:** Haneen Altaie  
**Program:** AI Integration and Governance, Humber College  

This project predicts Airbnb listing prices using machine learning based on listing features such as beds, guest ratings, reviews, property type, amenities, and listing details.

---

## Project Objective

The goal of this project was to build a machine learning model that can estimate the nightly price of an Airbnb listing.

---

## Tools and Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- AWS SageMaker
- Amazon S3
- Jupyter Notebook

---

## Dataset Overview

This project used Airbnb listing data containing information such as:

- number of beds
- guest ratings
- number of reviews
- listing title and description
- property type
- price and offer price

The dataset was cleaned and transformed to improve model performance.

---

## Feature Engineering

Several useful features were created from the raw Airbnb data, including:

- `beds`
- `rating`
- `num_reviews`
- `discount_pct`
- `detail_length`
- `is_luxury`
- `has_hot_tub`
- `has_pool`
- `has_pets`
- `is_treehouse`
- `property_type`

These features helped the model learn more meaningful pricing patterns.

---

## Models Tested

The following regression models were explored:

- Random Forest Regressor
- Gradient Boosting Regressor
- HistGradientBoosting Regressor
- XGBoost

---

## Final Best Model

The best-performing model was:

## **Random Forest v3**

This final version was improved by:

- removing extreme price outliers
- applying log transformation to the target variable
- engineering better features from raw text and listing details

---

## Final Results

| Metric | Result |
|--------|--------|
| MAE | **$55.56** |
| RMSE | **$74.84** |
| R² | **0.198** |

### Improvement from baseline
- MAE improved from **$74.70 → $55.56**
- RMSE improved from **$126.17 → $74.84**

This showed a strong reduction in prediction error compared to the original baseline model.

---

## AWS SageMaker Deployment

This project was also deployed using **Amazon SageMaker**.

Deployment workflow included:

- uploading data to **Amazon S3**
- training the model in **SageMaker Studio**
- saving the trained model using **joblib**
- packaging the model as `model.tar.gz`
- writing a custom `inference.py` script
- deploying the model as a **real-time endpoint**

---

## Deployment Challenges Solved

During deployment, several real-world ML engineering issues were resolved:

- column naming / `KeyError`
- scikit-learn version mismatch
- unsupported framework version string
- memory issues with endpoint instance type

These troubleshooting steps were an important part of the project because they reflect practical machine learning deployment work.

---

## Project Files

- `airbnb-sagemaker.ipynb` → main notebook
- `airbnb_final_report.pdf` → final written report
- `inference.py` → custom inference script for deployment
- `sagemaker_notebook_screenshots.pdf` → supporting screenshots from SageMaker workflow

---

## Key Learning Outcomes

Through this project, I practiced:

- machine learning model development
- feature engineering
- regression model evaluation
- cloud-based ML deployment
- debugging deployment and version compatibility issues
- working with AWS SageMaker in a real workflow

---

## Conclusion

This project demonstrates a full end-to-end machine learning workflow:

**data cleaning → feature engineering → model training → evaluation → deployment**

It was built not only to improve prediction performance, but also to gain hands-on experience with real-world cloud deployment and machine learning operations.
