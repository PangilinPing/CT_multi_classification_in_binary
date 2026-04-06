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
    f1_score,
    roc_curve,
    precision_recall_curve
)


# ===============================
# 路徑設定
# ===============================

DATASET_DIR = "dataset/NB15_small"

TRAIN_PATH = os.path.join(DATASET_DIR, "train.csv")
TEST_PATH  = os.path.join(DATASET_DIR, "test.csv")

RESULT_DIR = "rf_results"
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


def get_shap_class1_values(explainer, X):
    """
    相容不同 shap 版本
    回傳 binary class=1 的 SHAP 值，shape = (n_samples, n_features)
    """
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        # 舊版 binary classifier: [class0, class1]
        if len(shap_values) == 2:
            return np.array(shap_values[1])
        return np.array(shap_values[0])

    shap_values = np.array(shap_values)

    # 有些版本會回傳 (n_samples, n_features, n_classes)
    if shap_values.ndim == 3:
        return shap_values[:, :, 1]

    return shap_values


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
# Save sanitized dataset
# ===============================

print("Saving sanitized datasets...")

train_save = X_train.copy()
train_save["label"] = y_train
train_save.to_csv(
    os.path.join(RESULT_DIR, "train_sanitized.csv"),
    index=False
)

test_save = X_test.copy()
test_save["label"] = y_test
test_save.to_csv(
    os.path.join(RESULT_DIR, "test_sanitized.csv"),
    index=False
)


# ===============================
# Train RF
# ===============================

print("Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)


# ===============================
# Testing
# ===============================

print("Testing...")

test_scores = rf.predict_proba(X_test)[:, 1]
y_pred = (test_scores > 0.5).astype(int)

metrics, cm = evaluate(y_test, y_pred, test_scores)

print(metrics)


# ===============================
# Save prediction result
# ===============================

print("Saving prediction results...")

pred_result_df = X_test.copy()
pred_result_df["label"] = y_test
pred_result_df["pred"] = y_pred
pred_result_df["score"] = test_scores

pred_result_df.to_csv(
    os.path.join(RESULT_DIR, "test_prediction_results.csv"),
    index=False
)


# ===============================
# Save metrics
# ===============================

result_df = pd.DataFrame([metrics])
result_df.to_csv(
    os.path.join(RESULT_DIR, "rf_metrics.csv"),
    index=False
)


# ===============================
# Save feature importance
# ===============================

print("Saving feature importance...")

feature_importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

feature_importance_df.to_csv(
    os.path.join(RESULT_DIR, "rf_feature_importance.csv"),
    index=False
)

print(feature_importance_df.head(20))


# ===============================
# Save confusion matrix
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
    title="Random Forest Confusion Matrix"
)


# ===============================
# Save ROC / PR curve
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
# SHAP
# ===============================

print("Computing SHAP values...")

explainer = shap.TreeExplainer(rf)
shap_class1 = get_shap_class1_values(explainer, X_test)

if shap_class1.shape != X_test.shape:
    raise ValueError(
        f"SHAP shape mismatch: shap={shap_class1.shape}, X_test={X_test.shape}"
    )

# 1. 儲存每筆樣本的各特徵 shap 值
shap_df = pd.DataFrame(shap_class1, columns=X_test.columns)
shap_df.insert(0, "sample_index", np.arange(len(shap_df)))
shap_df["label"] = y_test
shap_df["pred"] = y_pred
shap_df["score"] = test_scores

shap_df.to_csv(
    os.path.join(RESULT_DIR, "test_shap_values.csv"),
    index=False,
    encoding="utf-8-sig"
)

# 2. 統計各特徵重要度
mean_abs_shap = np.abs(shap_class1).mean(axis=0)
mean_shap = shap_class1.mean(axis=0)
max_abs_shap = np.abs(shap_class1).max(axis=0)

shap_stats_df = pd.DataFrame({
    "feature": X_test.columns,
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

# 3. 存單一樣本的 shap 排序
sample_idx = 0
sample_shap_df = pd.DataFrame({
    "feature": X_test.columns,
    "shap_value": shap_class1[sample_idx],
    "abs_shap_value": np.abs(shap_class1[sample_idx]),
    "feature_value": X_test.iloc[sample_idx].values
}).sort_values("abs_shap_value", ascending=False)

sample_shap_df.to_csv(
    os.path.join(RESULT_DIR, f"sample_{sample_idx}_shap_sorted.csv"),
    index=False,
    encoding="utf-8-sig"
)

# 4. shap summary plot
plt.figure()
shap.summary_plot(
    shap_class1,
    X_test,
    show=False
)
plt.tight_layout()
plt.savefig(
    os.path.join(RESULT_DIR, "shap_summary.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()

# 5. shap bar plot
plt.figure()
shap.summary_plot(
    shap_class1,
    X_test,
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
print(os.path.join(RESULT_DIR, "train_sanitized.csv"))
print(os.path.join(RESULT_DIR, "test_sanitized.csv"))
print(os.path.join(RESULT_DIR, "test_prediction_results.csv"))
print(os.path.join(RESULT_DIR, "rf_metrics.csv"))
print(os.path.join(RESULT_DIR, "rf_feature_importance.csv"))
print(os.path.join(RESULT_DIR, "confusion_matrix.csv"))
print(os.path.join(RESULT_DIR, "confusion_matrix.png"))
print(os.path.join(RESULT_DIR, "roc_curve.png"))
print(os.path.join(RESULT_DIR, "pr_curve.png"))
print(os.path.join(RESULT_DIR, "test_shap_values.csv"))
print(os.path.join(RESULT_DIR, "shap_feature_statistics.csv"))
print(os.path.join(RESULT_DIR, f"sample_{sample_idx}_shap_sorted.csv"))
print(os.path.join(RESULT_DIR, "shap_summary.png"))
print(os.path.join(RESULT_DIR, "shap_bar.png"))