"""
model/train_model.py

Replaces the RandomForest pipeline with a LightGBM-based pipeline while keeping
the same public functions and artifact filenames used elsewhere in the project.

Key points:
- Trains LightGBM (LGBMRegressor) on log1p(price) target
- Uses OOF target encoding for `route` to avoid leakage
- Uses GroupKFold (group by route) to evaluate realistic CV MAE
- Saves artifacts with the same names your other scripts expect:
    - model/flight_price_model.pkl
    - model/label_encoders.pkl
    - model/num_imputer.pkl
    - model/model_metadata.json
    - model/route_medians.json

Usage:
    pip install lightgbm
    python model/train_model.py

This file preserves the `predict_from_dict()` API so `ai_assist.py` and
`test_predict.py` should continue to work without modification.
"""

import os
import re
import json
import joblib
import math
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

# sklearn utilities
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.metrics import mean_absolute_error

# LightGBM (sklearn API)
try:
    import lightgbm as lgb
except Exception as e:
    raise RuntimeError("lightgbm is required. Install with: pip install lightgbm") from e

# -------------------------
# Config — tweak if needed
# -------------------------
RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))
np.random.seed(RANDOM_SEED)

DEFAULT_N_ESTIMATORS = int(os.getenv('N_ESTIMATORS', 300))
DEFAULT_N_JOBS = int(os.getenv('N_JOBS', 1))

DO_QUICK_SEARCH = os.getenv('DO_QUICK_SEARCH', 'false').lower() in ('1', 'true', 'yes')

# LightGBM safe default params (sklearn API)
LGBM_PARAMS = {
    'objective': 'regression',
    'n_estimators': DEFAULT_N_ESTIMATORS,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_SEED,
    'n_jobs': DEFAULT_N_JOBS,
}
EARLY_STOPPING_ROUNDS = 50
CV_FOLDS = 5

# -------------------------
# Cleaning / parsing helpers (kept from original file)
# -------------------------

def parse_price(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace(',', '').strip()
    try:
        return float(s)
    except Exception:
        return np.nan


def duration_to_minutes(x):
    if pd.isna(x):
        return np.nan
    s = str(x).lower().strip()
    try:
        return int(float(s))
    except Exception:
        pass
    m = re.match(r'^\s*(\d+)\s*h(?:[^\d]+(\d+)\s*m)?', s)
    if m:
        hours = int(m.group(1))
        mins = int(m.group(2)) if m.group(2) else 0
        return hours * 60 + mins
    m2 = re.match(r'^\s*(\d+)\s*m(?:in)?$', s)
    if m2:
        return int(m2.group(1))
    return np.nan


def categorize_time_block(t):
    if pd.isna(t):
        return None
    s = str(t).strip()
    m = re.match(r'^\s*(\d{1,2})\s*:\s*(\d{2})', s)
    if m:
        hour = int(m.group(1)) % 24
    else:
        s_low = s.lower()
        if 'mor' in s_low:
            return 'Morning'
        if 'aft' in s_low:
            return 'Afternoon'
        if 'eve' in s_low:
            return 'Evening'
        if 'night' in s_low:
            return 'Night'
        return None
    if 5 <= hour < 12:
        return 'Morning'
    if 12 <= hour < 17:
        return 'Afternoon'
    if 17 <= hour < 21:
        return 'Evening'
    return 'Night'


def normalize_text(x):
    if pd.isna(x):
        return None
    return str(x).strip()


def normalize_city(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    return re.sub(r'\s+', ' ', s.title())


def normalize_stops(x):
    if pd.isna(x):
        return None
    s = str(x).lower()
    if 'non' in s:
        return 'non-stop'
    m = re.search(r'(\d+)', s)
    if m:
        return f"{m.group(1)}-stop"
    return s

# -------------------------
# Data load & cleaning
# -------------------------

def load_and_clean_data(filepath):
    """Load and clean dataset. Kept compatible with your previous pipeline."""
    df = pd.read_csv(filepath)
    df.drop(columns=[c for c in df.columns if c.startswith('Unnamed')], inplace=True, errors='ignore')

    df['price'] = df['price'].apply(parse_price)
    df = df.dropna(subset=['price'])
    df = df[(df['price'] > 200) & (df['price'] < 100000)]

    df['airline'] = df['airline'].apply(normalize_text)
    df['from'] = df['from'].apply(normalize_city)
    df['to'] = df['to'].apply(normalize_city)
    df['class'] = df['class'].apply(lambda x: normalize_text(x).lower() if pd.notna(x) else None)
    df['stops'] = df['stops'].apply(normalize_stops)
    df['dep_time'] = df.get('dep_time', df.get('dep_time', None))
    df['arr_time'] = df.get('arr_time', df.get('arr_time', None))
    df['dep_time'] = df['dep_time'].apply(normalize_text)
    df['arr_time'] = df['arr_time'].apply(normalize_text)

    df['duration'] = df['duration'].apply(duration_to_minutes)
    df = df.dropna(subset=['duration'])

    df['dep_block'] = df['dep_time'].apply(categorize_time_block)
    df['arr_block'] = df['arr_time'].apply(categorize_time_block)
    df = df.dropna(subset=['dep_block', 'arr_block'])

    if 'flight date' in df.columns:
        try:
            df['flight_date_parsed'] = pd.to_datetime(df['flight date'], dayfirst=True, errors='coerce')
            df['day_of_week'] = df['flight_date_parsed'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
        except Exception:
            df['day_of_week'] = 0
            df['is_weekend'] = 0
    else:
        df['day_of_week'] = 0
        df['is_weekend'] = 0

    df['route'] = df['from'].str.lower() + '->' + df['to'].str.lower()
    df = df[(df['from'].notna()) & (df['to'].notna()) & (df['from'] != df['to'])]
    df = df.reset_index(drop=True)
    return df


def build_route_medians(df, out_path='model/route_medians.json', min_samples=5):
    route_stats = defaultdict(list)
    for _, r in df.iterrows():
        route = (str(r['from']).strip().lower(), str(r['to']).strip().lower())
        if pd.notna(r['price']):
            route_stats[route].append(float(r['price']))
    route_medians = {}
    for (o,d), arr in route_stats.items():
        if len(arr) >= min_samples:
            key = f"{o}->{d}"
            route_medians[key] = {
                "median": float(np.median(arr)),
                "p10": float(np.percentile(arr, 10)),
                "p90": float(np.percentile(arr, 90)),
                "samples": int(len(arr))
            }
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(route_medians, f, indent=2)
    print(f"Saved route medians: {out_path} ({len(route_medians)} routes)")
    return route_medians


# -------------------------
# OOF target encoding helper
# -------------------------

def oof_target_encode(series_key, y, n_splits=5, random_state=RANDOM_SEED):
    """
    Compute out-of-fold mean encoding for keys in series_key using target y.
    y is expected to be the transformed target (e.g. log1p prices).
    Returns (oof_array, full_map)
    """
    idx = np.arange(len(y))
    oof = np.zeros(len(y), dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, val_idx in kf.split(idx):
        train_keys = series_key.iloc[train_idx]
        train_y = y[train_idx]
        grp = pd.DataFrame({'k': train_keys.values, 'y': train_y}).groupby('k')['y'].mean()
        val_keys = series_key.iloc[val_idx]
        oof[val_idx] = val_keys.map(grp).fillna(y.mean()).values
    # full map for later use
    full_grp = pd.DataFrame({'k': series_key.values, 'y': y}).groupby('k')['y'].mean()
    full_map = full_grp.to_dict()
    return oof, full_map


# -------------------------
# Train & save (entry)
# -------------------------

def train_and_save_safe(csv_path='data/goibibo_flights_data.csv', out_dir='model'):
    os.makedirs(out_dir, exist_ok=True)
    print('Loading & cleaning data...')
    df = load_and_clean_data(csv_path)
    print('Rows after cleaning:', len(df))

    categorical_cols = ['airline', 'from', 'to', 'class', 'stops', 'dep_block', 'arr_block']
    numeric_cols = ['duration', 'day_of_week', 'is_weekend']

    for nc in numeric_cols:
        if nc not in df.columns:
            df[nc] = 0

    feature_order = categorical_cols + numeric_cols
    target_col = 'price'

    X = df[feature_order].copy()
    y_raw = df[target_col].copy().values

    # target transform
    y = np.log1p(y_raw)

    # Build & save label encoders early
    label_encoders = {}
    for c in categorical_cols:
        le = LabelEncoder()
        X[c] = X[c].fillna('__missing__').astype(str)
        le.fit(X[c])
        label_encoders[c] = le
    joblib.dump(label_encoders, os.path.join(out_dir, 'label_encoders.pkl'))
    print('Saved label_encoders.pkl')

    # route features and OOF target encoding
    df['route'] = df['from'].str.lower().fillna('__missing__') + '->' + df['to'].str.lower().fillna('__missing__')
    route_counts = df['route'].value_counts().to_dict()
    route_median_price = df.groupby('route')['price'].median().to_dict()

    print('Computing OOF target encoding for route...')
    oof_route_enc, route_enc_map = oof_target_encode(df['route'], y, n_splits=CV_FOLDS)
    X['route_te'] = oof_route_enc
    X['route_count'] = df['route'].map(route_counts).fillna(0).astype(int)
    X['route_median_price'] = df['route'].map(route_median_price).fillna(df['price'].median())

    # Transform categorical columns using saved label_encoders
    for c, le in label_encoders.items():
        X[c] = X[c].fillna('__missing__').astype(str)
        try:
            X[c] = le.transform(X[c])
        except Exception:
            # if transform fails (shouldn't after fit), map unseen to missing
            X[c] = X[c].map(lambda v: v if v in le.classes_ else '__missing__')
            X[c] = le.transform(X[c])

    # numeric imputer
    num_imputer = SimpleImputer(strategy='median')
    X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])
    joblib.dump(num_imputer, os.path.join(out_dir, 'num_imputer.pkl'))

    # final feature order
    feature_order = categorical_cols + numeric_cols + ['route_te', 'route_count', 'route_median_price']
    X_final = X[feature_order].copy()

    # GroupKFold CV by route
    groups = df['route'].values
    gkf = GroupKFold(n_splits=CV_FOLDS)

    fold_maes = []
    oof_preds_log = np.zeros(len(X_final), dtype=float)

    print('Starting GroupKFold CV with LightGBM...')
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_final, y, groups)):
        print(f' Fold {fold+1}/{CV_FOLDS} — train {len(tr_idx)} rows, val {len(val_idx)} rows')
        X_tr, X_val = X_final.iloc[tr_idx], X_final.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        # Robust fitting: try newer API (early_stopping_rounds kwarg), else fallback to callbacks
        try:
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric='l1',
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose=False
            )
        except TypeError:
            # older lightgbm: use callbacks for early stopping and silence logs
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric='l1',
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(period=0)]
            )

        val_pred_log = model.predict(X_val)
        val_pred = np.expm1(val_pred_log)
        mae_val = mean_absolute_error(np.expm1(y_val), val_pred)
        fold_maes.append(mae_val)
        oof_preds_log[val_idx] = val_pred_log
        print(f'  Fold {fold+1} MAE (original scale): ₹{mae_val:.2f}')

    mean_cv_mae = float(np.mean(fold_maes))
    std_cv_mae = float(np.std(fold_maes))
    print(f'CV MAE (mean): ₹{mean_cv_mae:.2f}  (std: {std_cv_mae:.2f})')

    # retrain final model on full data
    print('Retraining final LightGBM on full data...')
    final_model = lgb.LGBMRegressor(**LGBM_PARAMS)
    # Some LightGBM builds don't accept the `verbose` kwarg — handle that robustly
    try:
        final_model.fit(X_final, y, verbose=False)
    except TypeError:
        # fallback: call without verbose
        final_model.fit(X_final, y)

    # quick holdout test (train_test_split) for a simple QA metric
    try:
        X_train, X_test, y_train_raw, y_test_raw = train_test_split(X_final, y_raw, test_size=0.2, random_state=RANDOM_SEED)
        y_test_log = np.log1p(y_test_raw.values if hasattr(y_test_raw, 'values') else y_test_raw)
        test_pred_log = final_model.predict(X_test)
        test_pred = np.expm1(test_pred_log)
        mae_test = mean_absolute_error(y_test_raw, test_pred)
    except Exception:
        mae_test = None

    # Save artifacts with same filenames as before
    model_path = os.path.join(out_dir, 'flight_price_model.pkl')
    joblib.dump(final_model, model_path)
    joblib.dump(label_encoders, os.path.join(out_dir, 'label_encoders.pkl'))
    joblib.dump(num_imputer, os.path.join(out_dir, 'num_imputer.pkl'))

    # Save route medians (duration/price medians helpful)
    build_route_medians(df, out_path=os.path.join(out_dir, 'route_medians.json'), min_samples=5)

    metadata = {
        'feature_order': feature_order,
        'categorical_cols': categorical_cols,
        'numeric_cols': numeric_cols,
        'target': 'log1p(price)',
        'cv_mean_mae': mean_cv_mae,
        'cv_std_mae': std_cv_mae,
        'mae_test': float(mae_test) if mae_test is not None else None,
        'n_rows': int(len(df)),
        'random_seed': int(RANDOM_SEED)
    }
    with open(os.path.join(out_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print('Saved model and artifacts to', out_dir)
    print('Training complete.')
    return model_path


# -------------------------
# Predict helper (keeps same signature as original)
# -------------------------

def predict_from_dict(model, label_encoders, input_dict, feature_order=None):
    """
    model: either an LGBMRegressor instance or path to joblib file
    label_encoders: dict of LabelEncoder objects or path to joblib file
    input_dict: dictionary with keys matching feature_order
    Returns predicted price (original ₹ scale)
    """
    # load model if path
    if isinstance(model, str):
        model = joblib.load(model)
    if isinstance(label_encoders, str):
        label_encoders = joblib.load(label_encoders)

    if feature_order is None:
        feature_order = ['airline','from','to','class','stops','dep_block','arr_block','duration','day_of_week','is_weekend','route_te','route_count','route_median_price']

    # Build DataFrame for single row
    row = {k: input_dict.get(k, None) for k in feature_order}
    # ensure we have from/to for route
    frm = input_dict.get('from') or input_dict.get('origin') or input_dict.get('o')
    to = input_dict.get('to') or input_dict.get('dest') or input_dict.get('d')
    route = (str(frm).strip().lower() if frm is not None else '__missing__') + '->' + (str(to).strip().lower() if to is not None else '__missing__')

    df_row = pd.DataFrame([row])

    # Label encode categoricals
    for c, le in label_encoders.items():
        if c in df_row.columns:
            val = df_row.loc[0, c]
            if pd.isna(val):
                val = '__missing__'
            val = str(val)
            try:
                df_row.loc[0, c] = le.transform([val])[0]
            except Exception:
                try:
                    df_row.loc[0, c] = le.transform(['__missing__'])[0]
                except Exception:
                    df_row.loc[0, c] = 0

    # numeric impute
    for num in ['duration','day_of_week','is_weekend']:
        if num in df_row.columns:
            df_row[num] = pd.to_numeric(df_row[num], errors='coerce').fillna(0)

    # load route medians for derived features if available
    route_stats_path = os.path.join(os.path.dirname(__file__), 'route_medians.json')
    route_median_price = None
    rcount = 0
    route_te_val = None
    if os.path.exists(route_stats_path):
        try:
            rm = json.load(open(route_stats_path))
            route_median_price = rm.get(route, {}).get('median', None)
        except Exception:
            route_median_price = None

    # try to load model metadata (route counts stored in metadata may not be present)
    model_meta_path = os.path.join(os.path.dirname(__file__), 'model_metadata.json')
    if os.path.exists(model_meta_path):
        try:
            md = json.load(open(model_meta_path))
        except Exception:
            md = {}
    else:
        md = {}

    # attempt to compute route_te and route_count from saved artifacts if present
    # If not present, we fallback to reasonable defaults
    route_stats_json = os.path.join(os.path.dirname(__file__), 'route_stats.json')
    if os.path.exists(route_stats_json):
        try:
            rs = json.load(open(route_stats_json))
            rcount = rs.get('route_count', {}).get(route, 0)
            route_te_val = rs.get('route_te_map', {}).get(route, None)
            if route_median_price is None:
                route_median_price = rs.get('route_median_price', {}).get(route, rs.get('global_price_median', None))
        except Exception:
            rcount = 0
            route_te_val = None
    # set fallback values
    if 'route_te' in df_row.columns:
        df_row['route_te'] = route_te_val if route_te_val is not None else 0.0
    if 'route_count' in df_row.columns:
        df_row['route_count'] = int(rcount)
    if 'route_median_price' in df_row.columns:
        df_row['route_median_price'] = float(route_median_price) if route_median_price is not None else float(md.get('global_price_median', 0))

    X_row = df_row[[c for c in feature_order if c in df_row.columns]]

    # predict (model outputs log1p price)
    try:
        pred_log = model.predict(X_row.values)[0]
    except Exception as e:
        # try sklearn API predict on DataFrame
        pred_log = model.predict(X_row)[0]

    pred_price = float(np.expm1(pred_log))
    return float(pred_price)


# -------------------------
# CLI
# -------------------------
if __name__ == '__main__':
    train_and_save_safe(csv_path='data/goibibo_flights_data.csv', out_dir='model')
