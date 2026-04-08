# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
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

import ct_value.f1_statistic as f1
import ct_value.f3_mapping as f3


# ===============================
# 路徑設定
# ===============================

DATASET_DIR = "dataset/NB15_small"

TRAIN_PATH = os.path.join(DATASET_DIR, "train.csv")
VALID_PATH = os.path.join(DATASET_DIR, "val.csv")
TEST_PATH  = os.path.join(DATASET_DIR, "test.csv")

FEATURE_COUNT_DIR = "feature_count_fs_from_shap"
RESULT_ROOT = "feature_selection_from_shap_results"

os.makedirs(FEATURE_COUNT_DIR, exist_ok=True)
os.makedirs(RESULT_ROOT, exist_ok=True)


# ===============================
# 模型與對應 SHAP 檔案設定
# ===============================

SHAP_THRESHOLD = 0.01

MODEL_CONFIG = {
    "nb": {
        "type": "raw",
        "shap_csv": "nb_results/shap_feature_statistics.csv"
    },
    "rf": {
        "type": "raw",
        "shap_csv": "rf_results/shap_feature_statistics.csv"
    },
    "ct_nb": {
        "type": "ct",
        "shap_csv": "ct_nb_results/shap_feature_statistics.csv"
    },
    "ct_rf": {
        "type": "ct",
        "shap_csv": "ct_rf_results/shap_feature_importance.csv"
    },
    "ct_sum": {
        "type": "ct",
        "shap_csv": "ct_sum_results/ctsum_shap_feature_statistics.csv"
    }
}


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
    drop_cols = [c for c in ["label", "attack_cat"] if c in df.columns]
    return sanitize(df.drop(columns=drop_cols))


def evaluate(y_true, y_pred, y_score):
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


def load_selected_features_from_shap(shap_csv_path, available_columns, threshold=0.01):
    if not os.path.exists(shap_csv_path):
        raise FileNotFoundError(f"找不到 SHAP 統計檔: {shap_csv_path}")

    shap_df = pd.read_csv(shap_csv_path)

    if "feature" not in shap_df.columns:
        raise ValueError(f"{shap_csv_path} 缺少 feature 欄位")

    score_col = None
    for c in ["mean_abs_shap", "abs_shap", "shap_value", "importance"]:
        if c in shap_df.columns:
            score_col = c
            break

    if score_col is None:
        raise ValueError(
            f"{shap_csv_path} 找不到 SHAP 分數欄位，需包含其中之一: "
            f"mean_abs_shap / abs_shap / shap_value / importance"
        )

    shap_df = shap_df.copy()
    shap_df = shap_df[shap_df["feature"].isin(available_columns)]

    selected_df = shap_df[shap_df[score_col] > threshold].copy()
    selected_df = selected_df.sort_values(score_col, ascending=False)

    if selected_df.empty:
        fallback_df = shap_df.sort_values(score_col, ascending=False).head(1).copy()
        if fallback_df.empty:
            raise ValueError(f"{shap_csv_path} 中沒有任何可對應到資料欄位的特徵")
        selected_df = fallback_df

    selected_features = selected_df["feature"].tolist()
    return selected_features, selected_df, score_col


def save_selected_features_csv(selected_df, save_path):
    selected_df.to_csv(save_path, index=False, encoding="utf-8-sig")


def train_and_eval_raw_nb(X_train, y_train, X_valid, y_valid, X_test, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)

    valid_score = model.predict_proba(X_valid)[:, 1]
    best_threshold, best_fpr, best_tpr, best_j = find_best_threshold_by_youden(y_valid, valid_score)

    test_score = model.predict_proba(X_test)[:, 1]
    y_pred = (test_score >= best_threshold).astype(int)

    metrics = evaluate(y_test, y_pred, test_score)
    metrics["Best_Threshold"] = best_threshold
    metrics["Val_Best_FPR"] = best_fpr
    metrics["Val_Best_TPR"] = best_tpr
    metrics["Val_Best_J"] = best_j

    return y_pred, test_score, metrics


def train_and_eval_raw_rf(X_train, y_train, X_valid, y_valid, X_test, y_test):
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    valid_score = model.predict_proba(X_valid)[:, 1]
    best_threshold, best_fpr, best_tpr, best_j = find_best_threshold_by_youden(y_valid, valid_score)

    test_score = model.predict_proba(X_test)[:, 1]
    y_pred = (test_score >= best_threshold).astype(int)

    metrics = evaluate(y_test, y_pred, test_score)
    metrics["Best_Threshold"] = best_threshold
    metrics["Val_Best_FPR"] = best_fpr
    metrics["Val_Best_TPR"] = best_tpr
    metrics["Val_Best_J"] = best_j

    return y_pred, test_score, metrics


def train_and_eval_ct_nb(X_train, y_train, X_valid, y_valid, X_test, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)

    valid_score = model.predict_proba(X_valid)[:, 1]
    best_threshold, best_fpr, best_tpr, best_j = find_best_threshold_by_youden(y_valid, valid_score)

    test_score = model.predict_proba(X_test)[:, 1]
    y_pred = (test_score >= best_threshold).astype(int)

    metrics = evaluate(y_test, y_pred, test_score)
    metrics["Best_Threshold"] = best_threshold
    metrics["Val_Best_FPR"] = best_fpr
    metrics["Val_Best_TPR"] = best_tpr
    metrics["Val_Best_J"] = best_j

    return y_pred, test_score, metrics


def train_and_eval_ct_rf(X_train, y_train, X_valid, y_valid, X_test, y_test):
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    valid_score = model.predict_proba(X_valid)[:, 1]
    best_threshold, best_fpr, best_tpr, best_j = find_best_threshold_by_youden(y_valid, valid_score)

    test_score = model.predict_proba(X_test)[:, 1]
    y_pred = (test_score >= best_threshold).astype(int)

    metrics = evaluate(y_test, y_pred, test_score)
    metrics["Best_Threshold"] = best_threshold
    metrics["Val_Best_FPR"] = best_fpr
    metrics["Val_Best_TPR"] = best_tpr
    metrics["Val_Best_J"] = best_j

    return y_pred, test_score, metrics


def train_and_eval_ct_sum(X_valid_ct, y_valid, X_test_ct, y_test):
    valid_ct_sum = compute_ct_sum(X_valid_ct)
    test_ct_sum = compute_ct_sum(X_test_ct)

    # 你的規則：CTsum 越大越偏 0，所以要取負號後才是越大越偏 1
    valid_score = -valid_ct_sum
    test_score = -test_ct_sum

    best_threshold, best_fpr, best_tpr, best_j = find_best_threshold_by_youden(y_valid, valid_score)
    y_pred = (test_score >= best_threshold).astype(int)

    metrics = evaluate(y_test, y_pred, test_score)
    metrics["Best_Threshold"] = best_threshold
    metrics["Val_Best_FPR"] = best_fpr
    metrics["Val_Best_TPR"] = best_tpr
    metrics["Val_Best_J"] = best_j

    return y_pred, test_score, metrics, test_ct_sum


# ===============================
# 讀取資料
# ===============================

print("Loading datasets...")

train_df = sanitize(pd.read_csv(TRAIN_PATH))
valid_df = sanitize(pd.read_csv(VALID_PATH))
test_df  = sanitize(pd.read_csv(TEST_PATH))

for name, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
    if "label" not in df.columns:
        raise ValueError(f"{name}.csv 必須包含 label 欄位")

X_train_raw = get_feature_df(train_df)
y_train = train_df["label"].values

X_valid_raw = get_feature_df(valid_df)
y_valid = valid_df["label"].values

X_test_raw = get_feature_df(test_df)
y_test = test_df["label"].values

print("Raw train shape:", X_train_raw.shape)
print("Raw valid shape:", X_valid_raw.shape)
print("Raw test  shape:", X_test_raw.shape)


# ===============================
# CT mapping 準備
# ===============================

print("Computing CT statistics from full train...")

f1.statistic(train_df.copy(), out_dir=FEATURE_COUNT_DIR)

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

print("Mapping raw -> CT...")

X_train_ct, _ = f3.map_features_to_ct(X_train_raw.copy(), ct_table, return_time=True)
X_valid_ct, _ = f3.map_features_to_ct(X_valid_raw.copy(), ct_table, return_time=True)
X_test_ct, _ = f3.map_features_to_ct(X_test_raw.copy(), ct_table, return_time=True)

print("CT train shape:", X_train_ct.shape)
print("CT valid shape:", X_valid_ct.shape)
print("CT test  shape:", X_test_ct.shape)


# ===============================
# 跑模型
# ===============================

all_metrics_rows = []

for model_name, cfg in MODEL_CONFIG.items():
    print("\n" + "=" * 60)
    print(f"Running model: {model_name}")

    save_dir = os.path.join(RESULT_ROOT, model_name)
    os.makedirs(save_dir, exist_ok=True)

    model_type = cfg["type"]
    shap_csv_path = cfg["shap_csv"]

    if model_type == "raw":
        available_columns = X_train_raw.columns.tolist()
        selected_features, selected_df, score_col = load_selected_features_from_shap(
            shap_csv_path,
            available_columns,
            threshold=SHAP_THRESHOLD
        )

        save_selected_features_csv(
            selected_df,
            os.path.join(save_dir, "selected_features_from_shap.csv")
        )

        Xtr = X_train_raw[selected_features].copy()
        Xva = X_valid_raw[selected_features].copy()
        Xte = X_test_raw[selected_features].copy()

        print(f"Selected {len(selected_features)} raw features by {score_col} > {SHAP_THRESHOLD}")
        print(selected_features)

        if model_name == "nb":
            y_pred, test_score, metrics = train_and_eval_raw_nb(
                Xtr, y_train, Xva, y_valid, Xte, y_test
            )
        elif model_name == "rf":
            y_pred, test_score, metrics = train_and_eval_raw_rf(
                Xtr, y_train, Xva, y_valid, Xte, y_test
            )
        else:
            raise ValueError(f"未知 raw 模型: {model_name}")

        pred_df = Xte.copy()
        pred_df["label"] = y_test
        pred_df["pred"] = y_pred
        pred_df["score"] = test_score

    elif model_type == "ct":
        available_columns = X_train_ct.columns.tolist()
        selected_features, selected_df, score_col = load_selected_features_from_shap(
            shap_csv_path,
            available_columns,
            threshold=SHAP_THRESHOLD
        )

        save_selected_features_csv(
            selected_df,
            os.path.join(save_dir, "selected_features_from_shap.csv")
        )

        Xtr = X_train_ct[selected_features].copy()
        Xva = X_valid_ct[selected_features].copy()
        Xte = X_test_ct[selected_features].copy()

        print(f"Selected {len(selected_features)} CT features by {score_col} > {SHAP_THRESHOLD}")
        print(selected_features)

        if model_name == "ct_nb":
            y_pred, test_score, metrics = train_and_eval_ct_nb(
                Xtr, y_train, Xva, y_valid, Xte, y_test
            )
            pred_df = Xte.copy()
            pred_df["label"] = y_test
            pred_df["pred"] = y_pred
            pred_df["score"] = test_score

        elif model_name == "ct_rf":
            y_pred, test_score, metrics = train_and_eval_ct_rf(
                Xtr, y_train, Xva, y_valid, Xte, y_test
            )
            pred_df = Xte.copy()
            pred_df["label"] = y_test
            pred_df["pred"] = y_pred
            pred_df["score"] = test_score

        elif model_name == "ct_sum":
            y_pred, test_score, metrics, raw_ct_sum = train_and_eval_ct_sum(
                Xva, y_valid, Xte, y_test
            )
            pred_df = Xte.copy()
            pred_df["label"] = y_test
            pred_df["pred"] = y_pred
            pred_df["ct_sum"] = raw_ct_sum
            pred_df["score_for_label1"] = test_score

        else:
            raise ValueError(f"未知 ct 模型: {model_name}")

    else:
        raise ValueError(f"未知模型類型: {model_type}")

    metrics["model"] = model_name
    metrics["selected_feature_count"] = len(selected_features)
    metrics["shap_threshold"] = SHAP_THRESHOLD

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(
        os.path.join(save_dir, f"{model_name}_metrics.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    pred_df.to_csv(
        os.path.join(save_dir, "prediction_result.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    save_confusion_matrix_image(
        cm,
        os.path.join(save_dir, "confusion_matrix.png"),
        title=f"{model_name} Confusion Matrix"
    )

    save_roc_curve(
        y_test,
        test_score,
        os.path.join(save_dir, "roc_curve.png")
    )

    save_pr_curve(
        y_test,
        test_score,
        os.path.join(save_dir, "pr_curve.png")
    )

    all_metrics_rows.append(metrics)
    print(f"Saved results for {model_name} -> {save_dir}")


# ===============================
# 存總表
# ===============================

all_metrics_df = pd.DataFrame(all_metrics_rows)
all_metrics_df = all_metrics_df.sort_values("AUROC", ascending=False).reset_index(drop=True)

all_metrics_df.to_csv(
    os.path.join(RESULT_ROOT, "all_model_metrics.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\nAll done.")
print("Saved summary:", os.path.join(RESULT_ROOT, "all_model_metrics.csv"))