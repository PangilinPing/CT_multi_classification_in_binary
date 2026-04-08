# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
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
TEST_PATH = os.path.join(DATASET_DIR, "test.csv")

FEATURE_COUNT_DIR = "feature_count"
RESULT_DIR = "ct_sum_results"

os.makedirs(FEATURE_COUNT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ===============================
# 工具函式
# ===============================

def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def get_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [col for col in ["label", "attack_cat"] if col in df.columns]
    return sanitize(df.drop(columns=drop_cols))


def evaluate(y_true, y_pred, y_score):
    """
    y_score 必須符合：
    分數越大，越傾向 label = 1
    這裡會傳入 reversed_ct_sum = -ct_sum
    因為原始 CTsum 越大越傾向 0
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "AUROC": roc_auc_score(y_true, y_score),
        "AUPRC": average_precision_score(y_true, y_score),
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
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


def find_best_threshold_by_youden(y_true, y_score):
    """
    y_score 必須是「分數越大越傾向 label=1」
    會排除 roc_curve 回傳的 inf threshold
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true 只有單一類別，無法計算 ROC threshold。")

    if np.isnan(y_score).any() or np.isinf(y_score).any():
        raise ValueError("y_score 含有 NaN 或 inf，無法計算 ROC threshold。")

    if np.unique(y_score).shape[0] < 2:
        raise ValueError("y_score 全部相同，無法找到有效 threshold。")

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_scores = tpr - fpr

    finite_mask = np.isfinite(thresholds)
    if finite_mask.sum() == 0:
        raise ValueError("thresholds 全部都是 inf，無法找到有效 threshold。")

    valid_thresholds = thresholds[finite_mask]
    valid_fpr = fpr[finite_mask]
    valid_tpr = tpr[finite_mask]
    valid_j = j_scores[finite_mask]

    best_idx = np.argmax(valid_j)

    return (
        valid_thresholds[best_idx],
        valid_fpr[best_idx],
        valid_tpr[best_idx],
        valid_j[best_idx]
    )


def compute_ct_sum(ct_df: pd.DataFrame) -> np.ndarray:
    """
    每一列是一筆樣本，每一欄是一個特徵 CT 值
    axis=1 表示把同一筆樣本的所有特徵 CT 值加總
    """
    return ct_df.sum(axis=1).values


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
    """
    y_score 必須符合：分數越大越傾向 label=1
    """
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
    """
    y_score 必須符合：分數越大越傾向 label=1
    """
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
# CTsum 專用 SHAP 函式
# ===============================

def ctsum_model_output(X):
    """
    SHAP 解釋用的模型輸出函式
    這裡保留原始 CTsum 本身
    輸入: numpy array 或 DataFrame
    輸出: CTsum 分數
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    return np.sum(X, axis=1)


# ===============================
# 讀取資料
# ===============================

print("Loading dataset...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df = sanitize(train_df)
test_df = sanitize(test_df)

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)

if "label" not in train_df.columns or "label" not in test_df.columns:
    raise ValueError("train.csv / test.csv 必須包含 'label' 欄位。")


# ===============================
# 切分 train / val
# ===============================

print("Splitting train into train_sub / val...")

train_sub_df, val_df = train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df["label"],
    random_state=42
)

train_sub_df = train_sub_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

X_train = get_feature_df(train_sub_df)
y_train = train_sub_df["label"].values

X_val = get_feature_df(val_df)
y_val = val_df["label"].values

X_test = get_feature_df(test_df)
y_test = test_df["label"].values


# ===============================
# Step1: CT statistic
# ===============================

print("Computing CT statistic from training subset...")

f1.statistic(train_sub_df.copy(), out_dir=FEATURE_COUNT_DIR)


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

print("Mapping train -> CT...")
ct_train_df, _ = f3.map_features_to_ct(
    X_train.copy(),
    ct_table,
    return_time=True
)

print("Mapping val -> CT...")
ct_val_df, _ = f3.map_features_to_ct(
    X_val.copy(),
    ct_table,
    return_time=True
)

print("Mapping test -> CT...")
ct_test_df, _ = f3.map_features_to_ct(
    X_test.copy(),
    ct_table,
    return_time=True
)


# ===============================
# Step4: 計算 CTsum
# ===============================

print("Computing CTsum...")

train_ct_sum = compute_ct_sum(ct_train_df)
val_ct_sum = compute_ct_sum(ct_val_df)
test_ct_sum = compute_ct_sum(ct_test_df)

# 你的定義：
# CTsum 越大越傾向 0
# CTsum 越小越傾向 1
# 因此要轉成 sklearn 慣用方向：分數越大越傾向 1
val_score = -val_ct_sum
test_score = -test_ct_sum

print("Direction check:")
print("val ct_sum label=0 mean:", val_ct_sum[y_val == 0].mean() if np.sum(y_val == 0) > 0 else "N/A")
print("val ct_sum label=1 mean:", val_ct_sum[y_val == 1].mean() if np.sum(y_val == 1) > 0 else "N/A")
print("val score (-ct_sum) label=0 mean:", val_score[y_val == 0].mean() if np.sum(y_val == 0) > 0 else "N/A")
print("val score (-ct_sum) label=1 mean:", val_score[y_val == 1].mean() if np.sum(y_val == 1) > 0 else "N/A")


# ===============================
# Step5: validation 找最佳 threshold
# ===============================

print("Finding best threshold on validation set...")

best_threshold, best_fpr, best_tpr, best_j = find_best_threshold_by_youden(
    y_val,
    val_score
)

print(f"Best threshold = {best_threshold:.6f}")
print(f"Best J score   = {best_j:.6f}")
print(f"Val FPR        = {best_fpr:.6f}")
print(f"Val TPR        = {best_tpr:.6f}")


# ===============================
# Step6: test 評估
# ===============================

print("Evaluating on test set...")

# 注意：是對 test_score 判斷，不是原始 test_ct_sum
y_test_pred = (test_score >= best_threshold).astype(int)

metrics = evaluate(y_test, y_test_pred, test_score)
metrics["Best_Threshold"] = best_threshold

print("Test metrics:")
for k, v in metrics.items():
    print(f"{k}: {v}")


# ===============================
# Step7: 儲存 CT mapped datasets
# ===============================

print("Saving CT datasets...")

ct_train_save = ct_train_df.copy()
ct_train_save["ct_sum"] = train_ct_sum
ct_train_save["score_for_label1"] = -train_ct_sum
ct_train_save["label"] = y_train
ct_train_save.to_csv(
    os.path.join(RESULT_DIR, "ct_train.csv"),
    index=False
)

ct_val_save = ct_val_df.copy()
ct_val_save["ct_sum"] = val_ct_sum
ct_val_save["score_for_label1"] = val_score
ct_val_save["label"] = y_val
ct_val_save.to_csv(
    os.path.join(RESULT_DIR, "ct_val.csv"),
    index=False
)

ct_test_save = ct_test_df.copy()
ct_test_save["ct_sum"] = test_ct_sum
ct_test_save["score_for_label1"] = test_score
ct_test_save["label"] = y_test
ct_test_save["pred"] = y_test_pred
ct_test_save.to_csv(
    os.path.join(RESULT_DIR, "ct_test.csv"),
    index=False
)


# ===============================
# Step8: 儲存 metrics / threshold
# ===============================

metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(
    os.path.join(RESULT_DIR, "ct_sum_metrics.csv"),
    index=False
)

threshold_df = pd.DataFrame([{
    "best_threshold": best_threshold,
    "val_best_fpr": best_fpr,
    "val_best_tpr": best_tpr,
    "val_best_j": best_j
}])
threshold_df.to_csv(
    os.path.join(RESULT_DIR, "best_threshold.csv"),
    index=False
)


# ===============================
# Step9: 畫評估圖
# ===============================

print("Saving evaluation plots...")

cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])

save_confusion_matrix_image(
    cm,
    os.path.join(RESULT_DIR, "confusion_matrix.png"),
    title="CTsum Confusion Matrix"
)

# 這裡也要用 test_score，而不是原始 test_ct_sum
save_roc_curve(
    y_test,
    test_score,
    os.path.join(RESULT_DIR, "roc_curve.png")
)

save_pr_curve(
    y_test,
    test_score,
    os.path.join(RESULT_DIR, "pr_curve.png")
)


# ===============================
# Step10: SHAP for CTsum
# ===============================

print("Computing SHAP for CTsum...")

background_size = min(200, len(ct_train_df))
background_df = ct_train_df.sample(n=background_size, random_state=42)

explainer = shap.Explainer(
    ctsum_model_output,
    background_df
)

test_explain_df = ct_test_df.copy()
shap_exp = explainer(test_explain_df)
shap_values = np.array(shap_exp.values)

if shap_values.shape != test_explain_df.shape:
    raise ValueError(
        f"SHAP shape mismatch: shap={shap_values.shape}, X={test_explain_df.shape}"
    )

# 1. 儲存每筆資料各特徵 shap 值
shap_df = pd.DataFrame(shap_values, columns=test_explain_df.columns)
shap_df.insert(0, "sample_index", np.arange(len(shap_df)))
shap_df["label"] = y_test
shap_df["pred"] = y_test_pred
shap_df["ct_sum"] = test_ct_sum
shap_df["score_for_label1"] = test_score

shap_df.to_csv(
    os.path.join(RESULT_DIR, "ctsum_shap_values.csv"),
    index=False,
    encoding="utf-8-sig"
)

# 2. 統計特徵重要度
mean_abs_shap = np.abs(shap_values).mean(axis=0)
mean_shap = shap_values.mean(axis=0)
max_abs_shap = np.abs(shap_values).max(axis=0)

shap_stats_df = pd.DataFrame({
    "feature": test_explain_df.columns,
    "mean_abs_shap": mean_abs_shap,
    "mean_shap": mean_shap,
    "max_abs_shap": max_abs_shap
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

shap_stats_df.to_csv(
    os.path.join(RESULT_DIR, "ctsum_shap_feature_statistics.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\nTop 20 CTsum SHAP features:")
print(shap_stats_df.head(20))

# 3. 單一樣本解釋
sample_idx = 0
sample_shap_df = pd.DataFrame({
    "feature": test_explain_df.columns,
    "shap_value": shap_values[sample_idx],
    "abs_shap_value": np.abs(shap_values[sample_idx]),
    "feature_value": test_explain_df.iloc[sample_idx].values
}).sort_values("abs_shap_value", ascending=False)

sample_shap_df.to_csv(
    os.path.join(RESULT_DIR, f"ctsum_sample_{sample_idx}_shap_sorted.csv"),
    index=False,
    encoding="utf-8-sig"
)

# 4. Summary plot
plt.figure()
shap.summary_plot(
    shap_values,
    test_explain_df,
    show=False
)
plt.tight_layout()
plt.savefig(
    os.path.join(RESULT_DIR, "ctsum_shap_summary.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()

# 5. Bar plot
plt.figure()
shap.summary_plot(
    shap_values,
    test_explain_df,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig(
    os.path.join(RESULT_DIR, "ctsum_shap_bar.png"),
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
print(os.path.join(RESULT_DIR, "ct_val.csv"))
print(os.path.join(RESULT_DIR, "ct_test.csv"))
print(os.path.join(RESULT_DIR, "ct_sum_metrics.csv"))
print(os.path.join(RESULT_DIR, "best_threshold.csv"))
print(os.path.join(RESULT_DIR, "confusion_matrix.png"))
print(os.path.join(RESULT_DIR, "roc_curve.png"))
print(os.path.join(RESULT_DIR, "pr_curve.png"))
print(os.path.join(RESULT_DIR, "ctsum_shap_values.csv"))
print(os.path.join(RESULT_DIR, "ctsum_shap_feature_statistics.csv"))
print(os.path.join(RESULT_DIR, f"ctsum_sample_{sample_idx}_shap_sorted.csv"))
print(os.path.join(RESULT_DIR, "ctsum_shap_summary.png"))
print(os.path.join(RESULT_DIR, "ctsum_shap_bar.png"))