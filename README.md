# Predicting Smartphone Addiction (Kaggle Playground Series - S6E8)

This repository contains the solution and code for the [Kaggle Playground Series - Season 6, Episode 8: Predicting Smartphone Addiction](https://www.kaggle.com/competitions/playground-series-s6e8) competition.

## Project Overview
The goal of this competition is to predict the addiction level of users based on their smartphone usage patterns, screen time, and other behavioral metrics. Our final approach relies on a robust stacking ensemble methodology utilizing a diverse set of models.

## Key Insights & Methodology

During this competition, we discovered two major pillars that drove our model's performance:

### 1. The Power of Tree-Based Models with Native NaN Handling
Missing values (NaNs) in this dataset contained significant predictive signal. Standard imputation techniques often destroyed this signal or introduced noise. We found that advanced tree-based models—specifically **LightGBM**, **XGBoost**, and **CatBoost**—performed exceptionally well because they can handle NaN features natively. By allowing the algorithms to determine the best splits for missing data inherently, we preserved the natural patterns and signals hidden within the dataset.

### 2. Generalization Beats Optimization (OOF Stacking)
While hyperparameter tuning a single model (like LightGBM) yields good results, we realized that **generalization beats the optimization of a single model**. 

Every algorithm has a unique mathematical approach and "way of thinking" when manipulating data. To capture these varied perspectives, we employed **Out-of-Fold (OOF) Stacking**:
- We trained a diverse set of base models (LightGBM, XGBoost, CatBoost, Random Forest, Extra Trees, SVM, MLP, Naive Bayes, etc.). We also developed a specialized **Masked Neural Network** in PyTorch that uses an input dropout (`mask_prob`) during training to robustly handle missing values (NaNs) by learning to treat zeros as missing representations.
- We generated Out-of-Fold (OOF) predictions for the training set to prevent data leakage.
- A meta-model (Logistic Regression) was then trained on these OOF predictions to synthesize the strengths of each base model. To address distribution differences among the models' predictions, we passed the base predictions through **rank transformations** before stacking, which led to a highly robust and generalized final predictor, bringing our score up to **0.96589**.

## Project Structure
- **`notebooks/`**: Contains the core notebooks for the project.
  - `Data_Exploration.ipynb`: Exploratory Data Analysis (EDA) and hypothesis testing.
  - `native_nan_fixed_optuna.ipynb` & `native_nan_optuna_v2.ipynb`: Tree models utilizing Optuna tuning and native NaN handling.
  - `diverse_models_oof.ipynb` & `svm_oof.ipynb`: Generation of OOF predictions from various distinct base models.
  - `nn_mask_oof.ipynb`: A PyTorch Neural Network utilizing a masking layer to robustly learn from NaN values.
  - `oof_stacking_rank.ipynb` & `oof_stacking_logits.ipynb`: The final meta-model training and stacking ensemble using rank and logit transformations.
  - `tuned_parameters.json`: Stored optimal hyperparameter configurations.
- **`datasets/`**: Data directory (you will need to download the Kaggle dataset into this folder).

## Setup & Installation

To reproduce this environment locally, you must use a Python Virtual Environment (`venv`) to keep dependencies clean and isolated.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mounir19000/smartphone-addiction-prediction.git
   cd smartphone-addiction-prediction
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Create the virtual environment
   python3 -m venv venv
   
   # Activate it (Linux/macOS)
   source venv/bin/activate
   
   # Activate it (Windows)
   # venv\Scripts\activate
   ```

3. **Install the required libraries:**
   Make sure your virtual environment is active, then install the necessary data science libraries used in this project:
   ```bash
   pip install pandas numpy scikit-learn lightgbm xgboost catboost optuna jupyter torch papermill
   ```

4. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```
