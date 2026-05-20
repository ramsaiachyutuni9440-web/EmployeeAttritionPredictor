# Employee Attrition Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange) ![License](https://img.shields.io/badge/License-MIT-green) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A machine learning project that predicts whether an employee is likely to leave a company, using the IBM HR Analytics dataset. Built with Python and scikit-learn as part of my transition from IT production support into AI/ML engineering.

---

## Problem Statement

Employee attrition is costly — replacing a single employee can cost up to 2x their annual salary. This project builds a classification model that identifies at-risk employees based on factors like overtime, job satisfaction, and monthly income, helping HR teams take proactive action.

---

## Demo

![Feature Importance Chart](feature_importance.png)

> Top 10 features driving attrition — OverTime and MonthlyIncome are the strongest predictors.

---

## Project Structure

```
employee-attrition-predictor/
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv   # IBM HR dataset (from Kaggle)
│
├── notebooks/
│   └── attrition_analysis.ipynb                  # Full EDA + modelling notebook
│
├── attrition_predictor.py                         # Main Python script
├── attrition_model.pkl                            # Saved trained model
├── feature_importance.png                         # Visualisation output
├── requirements.txt                               # Dependencies
└── README.md
```

---

## Dataset

- **Source:** [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) on Kaggle
- **Rows:** 1,470 employee records
- **Columns:** 35 features (age, department, overtime, job satisfaction, monthly income, etc.)
- **Target variable:** `Attrition` — Yes (left) or No (stayed)
- **Class imbalance:** ~16% attrition (Yes) vs 84% retained (No)

---

## Approach

| Step | Description |
|------|-------------|
| 1. EDA | Explored distributions, attrition rates by department, overtime impact |
| 2. Preprocessing | Label encoding for categorical columns, dropped non-informative columns |
| 3. Train/Test Split | 80/20 split with `random_state=42` for reproducibility |
| 4. Model | Random Forest Classifier (`n_estimators=100`) |
| 5. Evaluation | Accuracy, Precision, Recall, F1-score, Confusion Matrix |
| 6. Insights | Feature importance plot to identify key attrition drivers |

---

## Results

| Metric | Score |
|--------|-------|
| Accuracy | ~86% |
| Precision (Attrition=Yes) | ~0.72 |
| Recall (Attrition=Yes) | ~0.68 |
| F1-Score (Attrition=Yes) | ~0.70 |

> **Key insight:** OverTime, MonthlyIncome, Age, and JobSatisfaction were the top drivers of attrition in this dataset.

---

## Key Findings

- Employees who work **overtime are 3x more likely** to leave
- Attrition is highest in the **Sales department**
- Employees with **low job satisfaction** and **fewer years at company** are at higher risk
- Monthly income below a certain threshold correlates strongly with leaving

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/your-username/employee-attrition-predictor.git
cd employee-attrition-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Download from [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) and place it in the `data/` folder.

**4. Run the script**
```bash
python attrition_predictor.py
```

---

## Requirements

```
pandas>=1.5
scikit-learn>=1.3
matplotlib>=3.6
seaborn>=0.12
joblib>=1.2
```

---

## What I Learned

- End-to-end ML pipeline: data loading → cleaning → encoding → training → evaluation
- Handling class imbalance in classification problems
- Interpreting precision vs recall trade-offs in a real business context
- Using feature importance to generate actionable business insights, not just model metrics

---

## Future Improvements

- [ ] Try XGBoost and compare performance with Random Forest
- [ ] Handle class imbalance using SMOTE oversampling
- [ ] Add a Streamlit web app for interactive predictions
- [ ] Experiment with hyperparameter tuning using GridSearchCV
- [ ] Add SHAP values for more detailed model explainability

---

## About Me

I'm an IT production support professional with 4 years of mainframe experience, transitioning into AI/ML engineering. This is one of several projects I'm building to demonstrate hands-on ML skills.

- LinkedIn: [your-linkedin-url]
- GitHub: [[your-github-url](https://github.com/ramsaiachyutuni9440-web)]

---

## License

This project is licensed under the MIT License. The IBM HR Analytics dataset is publicly available on Kaggle for educational use.
