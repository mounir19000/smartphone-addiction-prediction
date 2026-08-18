"""
Native NaN + Optuna v2: Extended Tuning
========================================
Same core approach as native_nan_fixed_optuna.ipynb but with:
- 50 Optuna trials (up from 15)
- 5-Fold CV (up from 3-Fold) for more robust evaluation
- Wider search space: added reg_alpha, reg_lambda, min_split_gain
- Higher n_estimators range (up to 1000)
- Multi-seed final prediction for stability
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import optuna
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 1. Load Data (train_original.csv — preserves NaNs)
# ============================================================
print("Loading data...")
train = pd.read_csv('../datasets/train_original.csv')
test = pd.read_csv('../datasets/test.csv')

X = train.drop(['id', 'addicted_label'], axis=1)
y = train['addicted_label']
X_test = test.drop(['id'], axis=1)

print(f"Train shape: {X.shape}, Test shape: {X_test.shape}")
print(f"Train NaN count: {X.isna().sum().sum()}, Test NaN count: {X_test.isna().sum().sum()}")

# ============================================================
# 2. Proper Ordinal Mappings (preserve natural order)
# ============================================================
stress_mapping = {'Low': 0, 'Medium': 1, 'High': 2, 'Unknown': -1}
impact_mapping = {'No': 0, 'Yes': 1, 'Unknown': -1}

def apply_mappings(df):
    df_out = df.copy()
    df_out['stress_level'] = df_out['stress_level'].map(stress_mapping).fillna(-1).astype(int)
    df_out['academic_work_impact'] = df_out['academic_work_impact'].map(impact_mapping).fillna(-1).astype(int)
    df_out['gender'] = df_out['gender'].fillna('Unknown').astype('category')
    return df_out

X_preprocessed = apply_mappings(X)
X_test_preprocessed = apply_mappings(X_test)

# ============================================================
# 3. Feature Engineering (same ratios as before)
# ============================================================
def add_features(df):
    df_out = df.copy()
    denom_screen = df_out['daily_screen_time_hours'].replace(0, 0.001)
    denom_notif = df_out['notifications_per_day'].replace(0, 0.001)

    df_out['social_media_ratio'] = df_out['social_media_hours'] / denom_screen
    df_out['gaming_ratio'] = df_out['gaming_hours'] / denom_screen
    df_out['work_study_ratio'] = df_out['work_study_hours'] / denom_screen
    df_out['app_opens_per_hour'] = df_out['app_opens_per_day'] / denom_screen
    df_out['notifications_to_opens_ratio'] = df_out['app_opens_per_day'] / denom_notif
    df_out['sleep_deficit'] = 8.0 - df_out['sleep_hours']
    return df_out

X_preprocessed = add_features(X_preprocessed)
X_test_preprocessed = add_features(X_test_preprocessed)

print(f"Features after engineering: {X_preprocessed.shape[1]}")

# ============================================================
# 4. Optuna Hyperparameter Tuning — 50 trials, 5-Fold CV
# ============================================================
N_TRIALS = 50
N_FOLDS = 5

def objective(trial):
    params = {
        # Tree structure
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 200),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),

        # Sampling
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 5),

        # Regularization (NEW — these were missing before)
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),

        # Fixed params
        'is_unbalance': True,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1,
    }

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in cv.split(X_preprocessed, y):
        X_tr = X_preprocessed.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_val = X_preprocessed.iloc[val_idx]
        y_val = y.iloc[val_idx]

        model = LGBMClassifier(**params)
        model.fit(X_tr, y_tr)

        preds = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, preds))

    return np.mean(scores)


print(f"Starting Optuna tuning ({N_TRIALS} trials, {N_FOLDS}-Fold CV)...")
print("This will take a while — each trial trains 5 models on ~690k rows.\n")

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print(f"\n{'='*60}")
print(f"Best CV ROC-AUC: {study.best_value:.6f}")
print(f"Best Params: {study.best_params}")
print(f"{'='*60}\n")

# ============================================================
# 5. Train Final Model(s) with Best Params — Multi-Seed
# ============================================================
# Using 3 different seeds and averaging predictions for stability
SEEDS = [42, 123, 2026]

best_params = study.best_params.copy()
best_params['is_unbalance'] = True
best_params['verbose'] = -1
best_params['n_jobs'] = -1

print(f"Training {len(SEEDS)} final models with seeds {SEEDS}...")

all_preds = []
for seed in SEEDS:
    best_params['random_state'] = seed
    model = LGBMClassifier(**best_params)
    model.fit(X_preprocessed, y)
    preds = model.predict_proba(X_test_preprocessed)[:, 1]
    all_preds.append(preds)
    print(f"  Seed {seed} done.")

# Average across seeds
final_preds = np.mean(all_preds, axis=0)

# ============================================================
# 6. Save Submission
# ============================================================
os.makedirs('../submissions', exist_ok=True)
submission = pd.DataFrame({'id': test['id'], 'addicted_label': final_preds})
submission.to_csv('../submissions/native_nan_optuna_v2.csv', index=False)
print(f"\nSubmission saved to submissions/native_nan_optuna_v2.csv")
print(f"Predictions range: [{final_preds.min():.6f}, {final_preds.max():.6f}]")
print(f"Predictions mean:  {final_preds.mean():.6f}")
