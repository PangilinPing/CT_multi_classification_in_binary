# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, accuracy_score, f1_score

import ct_value.f1_statistic as f1
import ct_value.f3_mapping as f3


# ===============================
# 路徑設定
# ===============================

DATASET_DIR = "dataset/NB15_n_minmax"

TRAIN_PATH = os.path.join(DATASET_DIR, "train.csv")
TEST_PATH  = os.path.join(DATASET_DIR, "test.csv")

FEATURE_COUNT_DIR = "feature_count"

RESULT_DIR = "ct_rf_results"
os.makedirs(RESULT_DIR, exist_ok=True)


# ===============================
# 工具
# ===============================

def sanitize(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df


def evaluate(y_true, y_pred, y_score):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "AUROC": roc_auc_score(y_true, y_score),
        "AUPRC": average_precision_score(y_true, y_score),
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "TNR": tn/(tn+fp),
        "TPR": tp/(tp+fn),
        "PPV": tp/(tp+fp),
        "NPV": tn/(tn+fn)
    }

    return metrics


# ===============================
# Load dataset
# ===============================

print("Loading dataset...")

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

X_train = sanitize(train_df.drop(columns=["label"]))
y_train = train_df["label"].values

X_test = sanitize(test_df.drop(columns=["label"]))
y_test = test_df["label"].values


# ===============================
# Step1: CT statistic (train)
# ===============================

print("Computing CT statistic...")

f1.statistic(train_df.copy(), out_dir=FEATURE_COUNT_DIR)


# ===============================
# Step2: Load ratio table
# ===============================

n_black = (y_train==1).sum()
n_white = (y_train==0).sum()

ratio = n_black / max(n_white,1)
black_more = True if n_black >= n_white else False

ct_table,_ = f3.load_ratio_table(
    FEATURE_COUNT_DIR,
    ratio,
    black_more,
    return_time=True
)


# ===============================
# Step3: CT mapping
# ===============================

print("Mapping train → CT")

ct_train_df,_ = f3.map_features_to_ct(
    X_train.copy(),
    ct_table,
    return_time=True
)

print("Mapping test → CT")

ct_test_df,_ = f3.map_features_to_ct(
    X_test.copy(),
    ct_table,
    return_time=True
)


# ===============================
# Save CT mapped dataset
# ===============================

print("Saving CT mapped datasets...")

ct_train_save = ct_train_df.copy()
ct_train_save["label"] = y_train

ct_test_save = ct_test_df.copy()
ct_test_save["label"] = y_test

ct_train_save.to_csv(
    os.path.join(RESULT_DIR, "ct_train.csv"),
    index=False
)

ct_test_save.to_csv(
    os.path.join(RESULT_DIR, "ct_test.csv"),
    index=False
)

print("CT mapped datasets saved.")


# ===============================
# Step4: RF training (CT features)
# ===============================

print("Training RF using CT features")

rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)

rf.fit(ct_train_df, y_train)


# ===============================
# Step5: Testing
# ===============================

print("Testing model")

test_scores = rf.predict_proba(ct_test_df)[:,1]
y_pred = (test_scores > 0.5).astype(int)

metrics = evaluate(y_test, y_pred, test_scores)

print(metrics)


# ===============================
# Step6: Save result
# ===============================

result_df = pd.DataFrame([metrics])

result_df.to_csv(
    os.path.join(RESULT_DIR, "ct_rf_metrics.csv"),
    index=False
)

print("Result saved:", os.path.join(RESULT_DIR, "ct_rf_metrics.csv"))