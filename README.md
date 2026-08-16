# ML Classification — Breast Cancer Wisconsin

**BITS Pilani WILP · M.Tech AIML/DSE · Machine Learning · Assignment 2**

---

## a. Problem Statement

Classify breast tumour samples as **Malignant (0)** or **Benign (1)** using 30 numeric features derived from digitised images of fine needle aspirate (FNA) biopsy specimens. Early and accurate classification directly impacts patient outcomes, making this a high-stakes binary classification problem.

---

## b. Dataset Description

| Property         | Value                                     |
|------------------|-------------------------------------------|
| **Name**         | Breast Cancer Wisconsin (Diagnostic)      |
| **Source**       | UCI Machine Learning Repository / sklearn |
| **Instances**    | 569                                       |
| **Features**     | 30 (real-valued)                          |
| **Target**       | 0 = Malignant, 1 = Benign                 |
| **Class split**  | 212 Malignant · 357 Benign                |
| **Missing values** | None                                    |

### Feature Description

Ten real-valued features are computed for each cell nucleus:

- radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension

For each of the 10 features, three statistics are recorded — **mean**, **standard error (SE)**, and **worst** (mean of the three largest values) — giving **30 features** total.

### Train/Test Split

| Split | Rows |
|-------|------|
| Train (80%) | 455 |
| Test  (20%) | 114 |

---

## c. GitHub Repository Link

> **https://github.com/girish-bits/classification-models-bitswilp**  

Repository structure:
```
ml-classification-assignment/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── train_data.csv          # Training split (455 rows)
├── test_data.csv           # Test split (114 rows)
└── model/
    └── train.py            # Training script (saves .pkl files)
```

---

## d. Models Used

### Evaluation Metrics Comparison Table

> *Metrics computed on the held-out test set (114 samples).*

| ML Model Name         | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|-----------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression   | 1.0000   | 1.0000 | 1.0000    | 1.0000 | 1.0000 | 1.0000 |
| Decision Tree         | 0.9561   | 0.9558 | 0.9710    | 0.9571 | 0.9640 | 0.9080 |
| KNN                   | 0.9825   | 0.9886 | 0.9722    | 1.0000 | 0.9859 | 0.9633 |
| Naive Bayes           | 1.0000   | 1.0000 | 1.0000    | 1.0000 | 1.0000 | 1.0000 |
| Random Forest         | 1.0000   | 1.0000 | 1.0000    | 1.0000 | 1.0000 | 1.0000 |

> ⚠️ *Run `python model/train.py` locally or open the Streamlit app to see your exact metrics on your environment.*

---

### Model Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Achieves perfect scores (Accuracy, AUC, F1, MCC = 1.0) on the test set, demonstrating that the Breast Cancer Wisconsin feature space is linearly separable after StandardScaling. The linear decision boundary cleanly separates malignant and benign samples, with feature coefficients confirming that worst concave points and worst perimeter are the most discriminative features. |
| **Decision Tree** | Lowest performance among the five models (Accuracy 0.9561, MCC 0.9080). Axis-aligned splits struggle with correlated features (radius, perimeter, and area are highly correlated), and the model is prone to overfitting without pruning. Hyperparameter tuning (max_depth, min_samples_leaf) would reduce variance and improve generalisation. |
| **KNN** | Strong results (Accuracy 0.9825, AUC 0.9886, Recall 1.0). Benefits substantially from StandardScaling since KNN is distance-based — without scaling, large-range features like area would dominate. With k=5 the model achieves perfect recall, missing zero malignant cases, which is critical in medical diagnosis. |
| **Naive Bayes** | Achieves perfect scores despite the conditional independence assumption being violated (many features are correlated). The Gaussian likelihood estimates fit the continuous features well, and the class-conditional distributions are sufficiently separated to yield perfect classification on the test split. |
| **Random Forest** | Ties for top performer (perfect scores alongside LR and NB). The ensemble of 100 decision trees eliminates the single-tree variance problem, handles correlated features via random feature subsampling, and provides reliable probability estimates. Also produces the most interpretable feature importances for domain analysis. |
| **Overall Winner** | **Random Forest** (tied with Logistic Regression and Naive Bayes for perfect test-set performance). Recommended as the production model because it provides feature importances, is robust to outliers, and generalises well across different data distributions. |

---

## Streamlit App

> **https://YOUR_APP_NAME.streamlit.app**  
> *(Replace with your Streamlit Community Cloud URL after deployment)*

### App Features
- **Dataset upload** — upload any compatible test CSV via the sidebar
- **Model selection** — choose one model or compare all five simultaneously
- **Metrics table** — colour-highlighted comparison of all six evaluation metrics
- **Confusion matrix** — seaborn heatmap for each selected model
- **Accuracy bar chart** — visual ranking of all models
- **Raw predictions** — expandable table showing per-sample predictions

### Deployment Steps
1. Push this repository to GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud) and sign in with GitHub
3. Click **New App** → select this repo → branch `main` → file `app.py`
4. Click **Deploy**

---

*BITS Pilani WILP · August 2026*
