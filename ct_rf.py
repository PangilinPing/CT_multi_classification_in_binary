# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    accuracy_score,
    f1_score
)

import ct_value.f1_statistic as f1
import ct_value.f3_mapping as f3


# ===============================
# 路徑設定
# ===============================

DATASET_DIR = "dataset/NB15_small"

TRAIN_PATH = os.path.join(DATASET_DIR, "train.csv")
TEST_PATH  = os.path.join(DATASET_DIR, "test.csv")

FEATURE_COUNT_DIR = "feature_count"
RESULT_DIR = "ct_rf_results"

os.makedirs(FEATURE_COUNT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ===============================
# 工具
# ===============================

def sanitize(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def evaluate(y_true, y_pred, y_score):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "AUROC": roc_auc_score(y_true, y_score),
        "AUPRC": average_precision_score(y_true, y_score),
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "TNR": safe_div(tn, tn + fp),
        "TPR": safe_div(tp, tp + fn),
        "PPV": safe_div(tp, tp + fp),
        "NPV": safe_div(tn, tn + fn),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }

    return metrics


def get_drop_cols(df):
    return [c for c in ["label", "attack_cat"] if c in df.columns]


def get_shap_class1_values(explainer, X):
    """
    兼容不同 shap 版本，回傳 binary class=1 的 SHAP 值
    預期輸出 shape = (n_samples, n_features)
    """
    shap_values = explainer.shap_values(X)

    # 舊版：binary classifier 常回傳 [class0, class1]
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            return np.array(shap_values[1])
        return np.array(shap_values[0])

    shap_values = np.array(shap_values)

    # 某些版本可能是 (n_samples, n_features, n_classes)
    if shap_values.ndim == 3:
        return shap_values[:, :, 1]

    # 若直接就是 (n_samples, n_features)
    return shap_values


# ===============================
# Load dataset
# ===============================

print("Loading dataset...")

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

train_df = sanitize(train_df)
test_df = sanitize(test_df)

if "label" not in train_df.columns or "label" not in test_df.columns:
    raise ValueError("train.csv 和 test.csv 都必須包含 label 欄位")

train_drop_cols = get_drop_cols(train_df)
test_drop_cols = get_drop_cols(test_df)

X_train = sanitize(train_df.drop(columns=train_drop_cols))
y_train = train_df["label"].values

X_test = sanitize(test_df.drop(columns=test_drop_cols))
y_test = test_df["label"].values

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)


# ===============================
# Step1: CT statistic (train)
# ===============================

print("Computing CT statistic...")

f1.statistic(train_df.copy(), out_dir=FEATURE_COUNT_DIR)


# ===============================
# Step2: Load ratio table
# ===============================

n_black = (y_train == 1).sum()
n_white = (y_train == 0).sum()

ratio = n_black / max(n_white, 1)
black_more = True if n_black >= n_white else False

ct_table, _ = f3.load_ratio_table(
    FEATURE_COUNT_DIR,
    ratio,
    black_more,
    return_time=True
)


# ===============================
# Step3: CT mapping
# ===============================

print("Mapping train -> CT")
ct_train_df, _ = f3.map_features_to_ct(
    X_train.copy(),
    ct_table,
    return_time=True
)

print("Mapping test -> CT")
ct_test_df, _ = f3.map_features_to_ct(
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
ct_train_save.to_csv(
    os.path.join(RESULT_DIR, "ct_train.csv"),
    index=False
)

ct_test_save = ct_test_df.copy()
ct_test_save["label"] = y_test
ct_test_save.to_csv(
    os.path.join(RESULT_DIR, "ct_test.csv"),
    index=False
)

print("CT mapped datasets saved.")


# ===============================
# Step4: RF training (CT features)
# ===============================

print("Training RF using CT features...")

rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(ct_train_df, y_train)


# ===============================
# Step5: Testing
# ===============================

print("Testing model...")

test_scores = rf.predict_proba(ct_test_df)[:, 1]
y_pred = (test_scores > 0.5).astype(int)

metrics = evaluate(y_test, y_pred, test_scores)
print("Metrics:", metrics)


# ===============================
# Step6: Save result
# ===============================

result_df = pd.DataFrame([metrics])
result_df.to_csv(
    os.path.join(RESULT_DIR, "ct_rf_metrics.csv"),
    index=False
)

print("Result saved:", os.path.join(RESULT_DIR, "ct_rf_metrics.csv"))


# ===============================
# Step7: SHAP analysis
# ===============================

print("Computing SHAP values...")

explainer = shap.TreeExplainer(rf)
shap_class1 = get_shap_class1_values(explainer, ct_test_df)

if shap_class1.shape != ct_test_df.shape:
    raise ValueError(
        f"SHAP shape mismatch: shap={shap_class1.shape}, X={ct_test_df.shape}"
    )


# ===============================
# Step8: Save per-sample per-feature SHAP values
# ===============================

print("Saving per-sample SHAP values...")

shap_df = pd.DataFrame(shap_class1, columns=ct_test_df.columns)
shap_df.insert(0, "sample_index", np.arange(len(shap_df)))
shap_df["label"] = y_test
shap_df["pred"] = y_pred
shap_df["score"] = test_scores

shap_df.to_csv(
    os.path.join(RESULT_DIR, "ct_test_shap_values.csv"),
    index=False
)

print("Saved:", os.path.join(RESULT_DIR, "ct_test_shap_values.csv"))


# ===============================
# Step9: Save global SHAP feature importance
# ===============================

print("Saving global SHAP feature importance...")

mean_abs_shap = np.abs(shap_class1).mean(axis=0)

importance_df = pd.DataFrame({
    "feature": ct_test_df.columns,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

importance_df.to_csv(
    os.path.join(RESULT_DIR, "shap_feature_importance.csv"),
    index=False
)

print("Saved:", os.path.join(RESULT_DIR, "shap_feature_importance.csv"))


# ===============================
# Step10: Save one sample explanation details
# ===============================

print("Saving sample-0 explanation details...")

sample_idx = 0

sample_shap_df = pd.DataFrame({
    "feature": ct_test_df.columns,
    "shap_value": shap_class1[sample_idx],
    "abs_shap_value": np.abs(shap_class1[sample_idx]),
    "feature_value": ct_test_df.iloc[sample_idx].values
}).sort_values("abs_shap_value", ascending=False)

sample_shap_df.to_csv(
    os.path.join(RESULT_DIR, f"sample_{sample_idx}_shap_sorted.csv"),
    index=False
)

sample_feature_df = pd.DataFrame({
    "feature": ct_test_df.columns,
    "feature_value": ct_test_df.iloc[sample_idx].values
}).sort_values("feature_value", ascending=False)

sample_feature_df.to_csv(
    os.path.join(RESULT_DIR, f"sample_{sample_idx}_feature_values.csv"),
    index=False
)

print("Saved sample explanation CSV files.")


# ===============================
# Step11: SHAP plots
# ===============================

print("Saving SHAP plots...")

# summary plot
plt.figure()
shap.summary_plot(
    shap_class1,
    ct_test_df,
    show=False
)
plt.tight_layout()
plt.savefig(
    os.path.join(RESULT_DIR, "shap_summary.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()

# bar plot
plt.figure()
shap.summary_plot(
    shap_class1,
    ct_test_df,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig(
    os.path.join(RESULT_DIR, "shap_bar.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()

print("Saved:")
print(os.path.join(RESULT_DIR, "shap_summary.png"))
print(os.path.join(RESULT_DIR, "shap_bar.png"))


# ===============================
# 完成
# ===============================

print("\nAll done.")
print("Saved files:")
print(os.path.join(RESULT_DIR, "ct_train.csv"))
print(os.path.join(RESULT_DIR, "ct_test.csv"))
print(os.path.join(RESULT_DIR, "ct_rf_metrics.csv"))
print(os.path.join(RESULT_DIR, "ct_test_shap_values.csv"))
print(os.path.join(RESULT_DIR, "shap_feature_importance.csv"))
print(os.path.join(RESULT_DIR, f"sample_{sample_idx}_shap_sorted.csv"))
print(os.path.join(RESULT_DIR, f"sample_{sample_idx}_feature_values.csv"))
print(os.path.join(RESULT_DIR, "shap_summary.png"))
print(os.path.join(RESULT_DIR, "shap_bar.png"))