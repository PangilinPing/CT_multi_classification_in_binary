# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_curve,
    precision_recall_curve
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
RESULT_DIR = "ct_nb_results"

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

    return metrics, cm


def save_confusion_matrix_image(cm, save_path, title="Confusion Matrix"):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Pred 0", "Pred 1"])
    plt.yticks(tick_marks, ["True 0", "True 1"])

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_roc_curve(y_true, y_score, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUROC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_pr_curve(y_true, y_score, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AUPRC = {auprc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ===============================
# Load dataset
# ===============================

print("Loading dataset...")

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

if "label" not in train_df.columns or "label" not in test_df.columns:
    raise ValueError("train.csv 與 test.csv 必須包含 label 欄位")

drop_cols_train = [c for c in ["label", "attack_cat"] if c in train_df.columns]
drop_cols_test = [c for c in ["label", "attack_cat"] if c in test_df.columns]

X_train = sanitize(train_df.drop(columns=drop_cols_train))
y_train = train_df["label"].values

X_test = sanitize(test_df.drop(columns=drop_cols_test))
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
# Step4: Naive Bayes training
# ===============================

print("Training Naive Bayes using CT features...")

nb = GaussianNB()
nb.fit(ct_train_df, y_train)


# ===============================
# Step5: Testing
# ===============================

print("Testing model...")

test_scores = nb.predict_proba(ct_test_df)[:, 1]
y_pred = (test_scores > 0.5).astype(int)

metrics, cm = evaluate(y_test, y_pred, test_scores)

print(metrics)


# ===============================
# Step6: Save prediction result
# ===============================

print("Saving prediction results...")

pred_result_df = ct_test_df.copy()
pred_result_df["label"] = y_test
pred_result_df["pred"] = y_pred
pred_result_df["score"] = test_scores

pred_result_df.to_csv(
    os.path.join(RESULT_DIR, "test_prediction_results.csv"),
    index=False
)


# ===============================
# Step7: Save metrics
# ===============================

result_df = pd.DataFrame([metrics])
result_df.to_csv(
    os.path.join(RESULT_DIR, "ct_nb_metrics.csv"),
    index=False
)


# ===============================
# Step8: Save confusion matrix
# ===============================

print("Saving confusion matrix...")

cm_df = pd.DataFrame(
    cm,
    index=["True_0", "True_1"],
    columns=["Pred_0", "Pred_1"]
)

cm_df.to_csv(
    os.path.join(RESULT_DIR, "confusion_matrix.csv"),
    index=True
)

save_confusion_matrix_image(
    cm,
    os.path.join(RESULT_DIR, "confusion_matrix.png"),
    title="GaussianNB Confusion Matrix"
)


# ===============================
# Step9: Save ROC / PR curve
# ===============================

print("Saving ROC / PR curves...")

save_roc_curve(
    y_test,
    test_scores,
    os.path.join(RESULT_DIR, "roc_curve.png")
)

save_pr_curve(
    y_test,
    test_scores,
    os.path.join(RESULT_DIR, "pr_curve.png")
)


# ===============================
# Step10: SHAP
# ===============================

print("Computing SHAP values for GaussianNB...")

# 背景資料抽樣，避免太慢
background_size = min(200, len(ct_train_df))
background_df = ct_train_df.sample(n=background_size, random_state=42)

# 測試資料若太大，可抽樣加速；這裡先全部使用
explain_df = ct_test_df.copy()

# GaussianNB 不是 tree model，使用 model agnostic explainer
explainer = shap.Explainer(
    nb.predict_proba,
    background_df
)

shap_exp = explainer(explain_df)

# shap_exp.values 可能是 (n_samples, n_features, n_classes)
shap_values = np.array(shap_exp.values)

if shap_values.ndim == 3:
    shap_class1 = shap_values[:, :, 1]
elif shap_values.ndim == 2:
    shap_class1 = shap_values
else:
    raise ValueError(f"Unexpected SHAP shape: {shap_values.shape}")

if shap_class1.shape != explain_df.shape:
    raise ValueError(
        f"SHAP shape mismatch: shap={shap_class1.shape}, X={explain_df.shape}"
    )

# 1. 儲存每筆樣本的各特徵 shap 值
shap_df = pd.DataFrame(shap_class1, columns=explain_df.columns)
shap_df.insert(0, "sample_index", np.arange(len(shap_df)))
shap_df["label"] = y_test
shap_df["pred"] = y_pred
shap_df["score"] = test_scores

shap_df.to_csv(
    os.path.join(RESULT_DIR, "test_shap_values.csv"),
    index=False,
    encoding="utf-8-sig"
)

# 2. 各特徵整體 shap 統計
mean_abs_shap = np.abs(shap_class1).mean(axis=0)
mean_shap = shap_class1.mean(axis=0)
max_abs_shap = np.abs(shap_class1).max(axis=0)

shap_stats_df = pd.DataFrame({
    "feature": explain_df.columns,
    "mean_abs_shap": mean_abs_shap,
    "mean_shap": mean_shap,
    "max_abs_shap": max_abs_shap
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

shap_stats_df.to_csv(
    os.path.join(RESULT_DIR, "shap_feature_statistics.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\nTop 20 SHAP features:")
print(shap_stats_df.head(20))

# 3. 單一樣本 shap 排序
sample_idx = 0
sample_shap_df = pd.DataFrame({
    "feature": explain_df.columns,
    "shap_value": shap_class1[sample_idx],
    "abs_shap_value": np.abs(shap_class1[sample_idx]),
    "feature_value": explain_df.iloc[sample_idx].values
}).sort_values("abs_shap_value", ascending=False)

sample_shap_df.to_csv(
    os.path.join(RESULT_DIR, f"sample_{sample_idx}_shap_sorted.csv"),
    index=False,
    encoding="utf-8-sig"
)

# 4. SHAP summary plot
plt.figure()
shap.summary_plot(
    shap_class1,
    explain_df,
    show=False
)
plt.tight_layout()
plt.savefig(
    os.path.join(RESULT_DIR, "shap_summary.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()

# 5. SHAP bar plot
plt.figure()
shap.summary_plot(
    shap_class1,
    explain_df,
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


# ===============================
# 完成
# ===============================

print("\nAll done.")
print("Saved files:")
print(os.path.join(RESULT_DIR, "ct_train.csv"))
print(os.path.join(RESULT_DIR, "ct_test.csv"))
print(os.path.join(RESULT_DIR, "test_prediction_results.csv"))
print(os.path.join(RESULT_DIR, "ct_nb_metrics.csv"))
print(os.path.join(RESULT_DIR, "confusion_matrix.csv"))
print(os.path.join(RESULT_DIR, "confusion_matrix.png"))
print(os.path.join(RESULT_DIR, "roc_curve.png"))
print(os.path.join(RESULT_DIR, "pr_curve.png"))
print(os.path.join(RESULT_DIR, "test_shap_values.csv"))
print(os.path.join(RESULT_DIR, "shap_feature_statistics.csv"))
print(os.path.join(RESULT_DIR, f"sample_{sample_idx}_shap_sorted.csv"))
print(os.path.join(RESULT_DIR, "shap_summary.png"))
print(os.path.join(RESULT_DIR, "shap_bar.png"))