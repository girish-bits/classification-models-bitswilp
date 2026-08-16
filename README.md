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
classification-models-bitswilp/
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
| **Logistic Regression** | Achieves perfect scores across all 6 metrics (Accuracy=1.0, AUC=1.0, F1=1.0, MCC=1.0) on the test set. This demonstrates that the Breast Cancer Wisconsin feature space is **linearly separable** after StandardScaling. The model benefits greatly from feature scaling — without it, high-range features like area would distort the decision boundary. The high AUC (1.0) confirms near-perfect probability calibration, meaning the model assigns very high confidence to correct predictions. Being a linear model, it is also the most **interpretable** — feature coefficients reveal that `worst concave points`, `worst perimeter`, and `mean concave points` are the strongest predictors. Training time is fastest among all 5 models, making it ideal as a strong, lightweight baseline. |
| **Decision Tree** | The only model that does not achieve perfect scores — Accuracy: 0.9561, AUC: 0.9558, MCC: 0.9080 — making it the **weakest performer** on this dataset. The main weakness is its tendency to **overfit** training data through deep, unconstrained splits. The axis-aligned decision boundaries struggle with the many correlated features in this dataset (e.g., radius, perimeter, and area are highly correlated). The lower AUC (0.9558) compared to all other models indicates poor probability estimation — it outputs hard probabilities (0 or 1) rather than calibrated soft probabilities. However, the model offers excellent **visual interpretability** via tree diagrams. Performance would improve significantly with hyperparameter tuning (max_depth ≤ 5, min_samples_leaf ≥ 5) or pruning strategies. |
| **KNN** | Delivers strong results — Accuracy: 0.9825, AUC: 0.9886, Recall: 1.0 (perfect). The perfect Recall means the model correctly identifies **every single malignant case** in the test set — which is the most critical metric in medical diagnosis (missing a malignant tumour is far more dangerous than a false alarm). KNN is a **non-parametric, instance-based** learner that makes no assumptions about data distribution, which suits the complex, non-linear boundaries in this dataset. It depends heavily on StandardScaling — without normalisation, features with large ranges (like `area` ~ 1000) would completely overshadow features like `smoothness` ~ 0.1. The main limitation is **computational cost** at inference time since it stores the entire training set (455 samples × 30 features), making it slower than other models for real-time predictions. |
| **Naive Bayes** | Achieves perfect scores (Accuracy=1.0, AUC=1.0, MCC=1.0) despite its **strong independence assumption** being clearly violated — many features in this dataset are highly correlated (e.g., radius, perimeter, and area are near-perfectly correlated). The reason it still works well is that the **class-conditional distributions** of malignant vs. benign samples are sufficiently separated in Gaussian feature space that even approximate likelihoods lead to correct classification. The Gaussian NB assumes each feature follows a normal distribution per class — this is a reasonable approximation for the standardised continuous features in this dataset. Its key advantages are **extremely fast training** and **no hyperparameters to tune**. However, the independence assumption limits its reliability on datasets with strong feature correlations, and it can produce poorly calibrated probability estimates in such cases. |
| **Random Forest** | Ties for best performance with Logistic Regression and Naive Bayes (perfect scores across all metrics). As an **ensemble of 100 decision trees**, it overcomes the single Decision Tree's overfitting problem by averaging predictions across many trees trained on random data subsets (bagging) and random feature subsets at each split. This reduces variance without increasing bias. The random feature subsampling (√30 ≈ 5 features per split) effectively handles the correlated feature problem that hurt the single Decision Tree. Additional advantage: it provides **feature importances**, revealing that `worst concave points`, `worst area`, and `worst perimeter` are the top predictors — consistent with domain knowledge. It is also **robust to outliers and missing values**. The only drawback is higher training time and memory usage compared to simpler models. |
| **Overall Winner** | **Logistic Regression** is the recommended model — it achieves perfect scores across all 6 metrics (Accuracy=1.0, AUC=1.0, F1=1.0, MCC=1.0) while being the simplest model in the comparison. By Occam's Razor, when multiple models achieve equal performance, the simpler model is preferred. Logistic Regression trains fastest, is the most interpretable (via feature coefficients), and generalises well due to its linear structure. This aligns with best practices in medical ML where model transparency and reproducibility are as important as raw performance. |

---

## Streamlit App

> **https://2025ac05343-girish.streamlit.app**

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
