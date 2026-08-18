# Future Ideas & Strategic Pivot

After analyzing the latest leaderboard scores and digging deeper into the datasets, we have uncovered a few critical insights that should drive our next iterations.

## 1. The Missing Data Breakthrough (Use `train_original.csv`)
* **The Discovery:** We noticed that `train.csv` has **zero** missing values, while `test.csv` has tens of thousands of missing values across almost every column. It turns out `train.csv` is just `train_original.csv` that someone has already pre-imputed!
* **Why Native NaNs Failed (0.916):** When we tried letting LightGBM handle NaNs natively (Idea 3), the score tanked. Why? Because we trained it on `train.csv`. The model never saw a single NaN during training, so it couldn't learn the optimal tree splits for missing data. When it finally saw NaNs in the `test.csv`, it had to guess, resulting in terrible performance.
* **The Fix:** We must switch our pipeline to train on `train_original.csv`. By doing this and dropping the `SimpleImputer`, LightGBM and XGBoost will natively learn the hidden signals behind *why* data is missing, which often provides a massive boost in Kaggle competitions.

## 2. Proper Ordinal Encoding
* **The Problem:** We are currently using `OrdinalEncoder` for `stress_level` and `academic_work_impact`. By default, this encodes categories alphabetically (e.g., High=0, Low=1, Medium=2). This completely destroys the natural ordinal relationship of the data!
* **The Fix:** We need to manually map these to integers to preserve their rank:
  * `stress_level`: `{'Low': 0, 'Medium': 1, 'High': 2, 'Unknown': -1}`
  * `academic_work_impact`: `{'No': 0, 'Yes': 1, 'Unknown': -1}`

## 3. Addressing Target Imbalance
* **The Problem:** The target variable `addicted_label` is imbalanced, with roughly 490k positive cases and 200k negative cases (~2.4:1 ratio).
* **The Fix:** We should experiment with setting `scale_pos_weight` or `is_unbalance=True` in LightGBM/XGBoost. While ROC-AUC is robust to class imbalance, properly calibrated probabilities can improve the performance of downstream Stacking meta-models.

## 4. True Out-of-Fold Stacking
* **The Problem:** Our ensembling script used a simple arithmetic average of LGBM, XGBoost, and CatBoost.
* **The Fix:** Instead of simple blending, we should implement a rigorous Stacking framework. We can generate out-of-fold (OOF) predictions for the train set using 5-Fold CV for heavily tuned LGBM, XGBoost, and CatBoost models. Then, we train a Logistic Regression or Ridge meta-model on those OOF predictions to learn exactly which base model to trust for different types of predictions.

## 5. Explicit Missing Indicators (For Neural Nets / Linear Models)
* If we want to include non-tree models (like TabNet or Logistic Regression) in our stack, we will still have to impute `train_original.csv`. However, we **must** set `add_indicator=True` in `SimpleImputer` so the models can still see that the data was originally missing.
