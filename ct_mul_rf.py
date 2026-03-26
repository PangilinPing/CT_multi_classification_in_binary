# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score
)

import ct_value.f1_statistic as f1
import ct_value.f3_mapping as f3


# ===============================
# 路徑
# ===============================
DATASET_DIR = "dataset/NB15"
TRAIN_PATH = os.path.join(DATASET_DIR, "train.csv")
TEST_PATH  = os.path.join(DATASET_DIR, "test.csv")

RESULT_DIR = "ct_dynamic_gini_global_ratio"
FC_DIR = os.path.join(RESULT_DIR, "feature_count")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FC_DIR, exist_ok=True)


# ===============================
# 工具
# ===============================
def sanitize(df):
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


def get_features(df):
    return [c for c in df.columns if c not in ["label", "attack_cat"]]


# ===============================
# Gini 計算
# ===============================
def compute_gini(series):
    values = np.abs(series.values.astype(float))
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return 0.0

    total = values.sum()
    if total == 0:
        return 0.0

    probs = values / total
    return 1.0 - np.sum(probs ** 2)


# ===============================
# Gini feature selection
# ===============================
def select_features_by_gini(X_ct, threshold=0.05, fallback_topk=20):
    gini_scores = {}

    for col in X_ct.columns:
        gini_scores[col] = compute_gini(X_ct[col])

    gini_series = pd.Series(gini_scores).sort_values(ascending=False)

    selected = gini_series[gini_series > threshold].index.tolist()

    # fallback：避免一個都沒有
    if len(selected) == 0:
        selected = gini_series.head(fallback_topk).index.tolist()

    return selected, gini_series


# ===============================
# 不做 1:2 sampling
# 改成用原始資料分布
# ===============================
def build_subset(df, attack=None):
    """
    attack=None:
        用整份資料做 global label 0/1
    attack=某 attack_cat:
        用 normal + 該 attack_cat 做二分類
    """
    if attack is None:
        subset = df.copy()
        y = subset["label"].values
    else:
        subset = df[(df["label"] == 0) | (df["attack_cat"] == attack)].copy()

        if len(subset) == 0:
            return None, None

        # 若該 attack 在 subset 中實際不存在，也視為無效
        if (subset["attack_cat"] == attack).sum() == 0:
            return None, None

        y = (subset["attack_cat"] == attack).astype(int).values

    return subset.reset_index(drop=True), y


# ===============================
# CT
# ===============================
def compute_ct(subset, y, name, global_ratio, global_black_more):
    save = os.path.join(FC_DIR, name)
    if os.path.exists(save):
        shutil.rmtree(save)
    os.makedirs(save)

    X = subset[get_features(subset)].copy()
    y_series = pd.Series(np.asarray(y).reshape(-1), name="label")

    df_ct = pd.concat([X, y_series], axis=1)

    f1.statistic(df_ct, out_dir=save)

    # 用整個 train 的黑白比例
    ct, _ = f3.load_ratio_table(
        save,
        global_ratio,
        global_black_more,
        return_time=True
    )

    return ct


def map_ct(df, ct):
    X = sanitize(df[get_features(df)].copy())
    ct_df, _ = f3.map_features_to_ct(X, ct, return_time=True)
    ct_df = sanitize(ct_df)
    return ct_df


# ===============================
# Dynamic CT
# 先用選到的特徵算 CT sum
# 再用該 mapping 的全部特徵當最終表示
# ===============================
def dynamic_ct_transform_full(df, ct_models, top_map):
    valid_ct = list(top_map.keys())

    mapped = {}
    for name in valid_ct:
        mapped[name] = map_ct(df, ct_models[name])

    final_features = []
    chosen_ct = []
    chosen_abs_sum = []
    chosen_raw_sum = []

    for i in range(len(df)):
        best_abs = -1
        best_ct = None
        best_sum = None

        for name in valid_ct:
            feats = top_map[name]
            vals = mapped[name].iloc[i][feats].values.astype(float)
            score = np.sum(vals)

            if abs(score) > best_abs:
                best_abs = abs(score)
                best_ct = name
                best_sum = score

        final_features.append(mapped[best_ct].iloc[i].values)
        chosen_ct.append(best_ct)
        chosen_abs_sum.append(best_abs)
        chosen_raw_sum.append(best_sum)

    feature_df = pd.DataFrame(final_features, columns=mapped[valid_ct[0]].columns)

    return feature_df, chosen_ct, chosen_raw_sum, chosen_abs_sum


# ===============================
# 畫 heatmap
# ===============================
def plot_crosstab_heatmap(ct_table, save_path, title):
    fig, ax = plt.subplots(figsize=(8, 6))

    arr = ct_table.values
    im = ax.imshow(arr, aspect="auto")

    ax.set_xticks(np.arange(ct_table.shape[1]))
    ax.set_yticks(np.arange(ct_table.shape[0]))
    ax.set_xticklabels(ct_table.columns)
    ax.set_yticklabels(ct_table.index)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for i in range(ct_table.shape[0]):
        for j in range(ct_table.shape[1]):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center")

    ax.set_title(title)
    ax.set_xlabel("Label")
    ax.set_ylabel("Chosen CT Mapping")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# ===============================
# 主程式
# ===============================
print("Loading...")

train = sanitize(pd.read_csv(TRAIN_PATH))
test  = sanitize(pd.read_csv(TEST_PATH))

train["label"] = train["label"].astype(int)
test["label"]  = test["label"].astype(int)

y_train = train["label"].values
y_test  = test["label"].values

# 整個 train 的黑白比例
global_n1 = int((train["label"] == 1).sum())
global_n0 = int((train["label"] == 0).sum())
global_ratio = global_n1 / max(global_n0, 1)
global_black_more = global_n1 >= global_n0

print(f"Global normal count : {global_n0}")
print(f"Global attack count : {global_n1}")
print(f"Global attack/normal ratio : {global_ratio:.6f}")


# ===============================
# Step1：建立 CT
# ===============================
ct_models = {}

# global CT
subset, y = build_subset(train, attack=None)
ct_models["global"] = compute_ct(
    subset, y, "global",
    global_ratio=global_ratio,
    global_black_more=global_black_more
)

# attack-specific CT
attacks = train.loc[train["label"] == 1, "attack_cat"].dropna().unique()

for atk in attacks:
    subset, y = build_subset(train, atk)

    if subset is None:
        print(f"Skip CT: atk_{atk}")
        continue

    name = f"atk_{atk}"
    ct_models[name] = compute_ct(
        subset, y, name,
        global_ratio=global_ratio,
        global_black_more=global_black_more
    )

print("Valid CT:", list(ct_models.keys()))


# ===============================
# Step2：Gini feature selection
# ===============================
top_map = {}
gini_score_df = []

for name, ct in ct_models.items():
    print("Selecting features (Gini > 0.05):", name)

    if name == "global":
        subset, y = build_subset(train, attack=None)
    else:
        atk = name.replace("atk_", "")
        try:
            atk = int(atk)
        except:
            pass
        subset, y = build_subset(train, atk)

    if subset is None:
        print(f"Skip feature selection: {name}")
        continue

    X_ct = map_ct(subset, ct)

    feats, gini_series = select_features_by_gini(
        X_ct,
        threshold=0.05,
        fallback_topk=20
    )

    top_map[name] = feats

    # 存每個 CT 的 gini 分數
    tmp = gini_series.reset_index()
    tmp.columns = ["feature", "gini"]
    tmp["ct_name"] = name
    gini_score_df.append(tmp)

    # 存每個 CT 被選到的特徵
    pd.DataFrame({"feature": feats}).to_csv(
        os.path.join(RESULT_DIR, f"{name}_selected_features.csv"),
        index=False
    )

if len(gini_score_df) > 0:
    pd.concat(gini_score_df, axis=0, ignore_index=True).to_csv(
        os.path.join(RESULT_DIR, "all_ct_gini_scores.csv"),
        index=False
    )

print("Final CT:", list(top_map.keys()))


# ===============================
# Step3：Dynamic CT
# ===============================
train_ct, train_choice, train_raw_sum, train_abs_sum = dynamic_ct_transform_full(
    train, ct_models, top_map
)
test_ct, test_choice, test_raw_sum, test_abs_sum = dynamic_ct_transform_full(
    test, ct_models, top_map
)

train_ct.to_csv(os.path.join(RESULT_DIR, "train_dynamic_ct_features.csv"), index=False)
test_ct.to_csv(os.path.join(RESULT_DIR, "test_dynamic_ct_features.csv"), index=False)


# ===============================
# Step4：模型
# ===============================
rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(train_ct, y_train)

pred_proba = rf.predict_proba(test_ct)[:, 1]
y_pred = (pred_proba > 0.5).astype(int)


# ===============================
# 評估
# ===============================
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

ACC = accuracy_score(y_test, y_pred)
AUROC = roc_auc_score(y_test, pred_proba)
AUPRC = average_precision_score(y_test, pred_proba)
F1 = f1_score(y_test, y_pred)

TPR = recall_score(y_test, y_pred, zero_division=0)
TNR = tn / (tn + fp) if (tn + fp) > 0 else 0
PPV = precision_score(y_test, y_pred, zero_division=0)
NPV = tn / (tn + fn) if (tn + fn) > 0 else 0

print("\n===== Metrics =====")
print(f"ACC   : {ACC:.6f}")
print(f"AUROC : {AUROC:.6f}")
print(f"AUPRC : {AUPRC:.6f}")
print(f"F1    : {F1:.6f}")
print(f"TPR   : {TPR:.6f}")
print(f"TNR   : {TNR:.6f}")
print(f"PPV   : {PPV:.6f}")
print(f"NPV   : {NPV:.6f}")
print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")


# ===============================
# 存 metrics / prediction
# ===============================
pd.DataFrame([{
    "ACC": ACC,
    "AUROC": AUROC,
    "AUPRC": AUPRC,
    "F1": F1,
    "TPR": TPR,
    "TNR": TNR,
    "PPV": PPV,
    "NPV": NPV,
    "TP": tp,
    "TN": tn,
    "FP": fp,
    "FN": fn
}]).to_csv(os.path.join(RESULT_DIR, "metrics.csv"), index=False)

pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred,
    "y_score": pred_proba,
    "chosen_ct": test_choice,
    "chosen_ct_sum": test_raw_sum,
    "chosen_abs_ct_sum": test_abs_sum
}).to_csv(os.path.join(RESULT_DIR, "predictions.csv"), index=False)

pd.DataFrame({
    "y_true": y_train,
    "chosen_ct": train_choice,
    "chosen_ct_sum": train_raw_sum,
    "chosen_abs_ct_sum": train_abs_sum
}).to_csv(os.path.join(RESULT_DIR, "train_row_mapping.csv"), index=False)


# ===============================
# CT usage
# ===============================
ct_count = pd.Series(test_choice).value_counts().sort_index()
ct_count_df = ct_count.reset_index()
ct_count_df.columns = ["ct_name", "count"]
ct_count_df.to_csv(os.path.join(RESULT_DIR, "ct_usage.csv"), index=False)

plt.figure(figsize=(10, 6))
ct_count.plot(kind="bar")
plt.title("CT Usage Count")
plt.xlabel("CT Mapping")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "ct_usage.png"))
plt.close()


# ===============================
# Feature usage
# ===============================
all_feats = []
for feat_list in top_map.values():
    all_feats.extend(feat_list)

feat_count = pd.Series(all_feats).value_counts()
feat_count_df = feat_count.reset_index()
feat_count_df.columns = ["feature", "count"]
feat_count_df.to_csv(os.path.join(RESULT_DIR, "feature_usage.csv"), index=False)

plt.figure(figsize=(12, 6))
feat_count.head(30).plot(kind="bar")
plt.title("Feature Usage Count (Gini > 0.05)")
plt.xlabel("Feature")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "feature_usage.png"))
plt.close()


# ===============================
# Row mapping confusion matrix / crosstab
# ===============================
row_map_df = pd.DataFrame({
    "label": y_test,
    "chosen_ct": test_choice
})

ct_vs_label = pd.crosstab(row_map_df["chosen_ct"], row_map_df["label"])
ct_vs_label.to_csv(os.path.join(RESULT_DIR, "ct_vs_label_confusion.csv"))

print("\nCT vs Label confusion:")
print(ct_vs_label)

# heatmap
plot_crosstab_heatmap(
    ct_vs_label,
    os.path.join(RESULT_DIR, "ct_vs_label_confusion_heatmap.png"),
    "Chosen CT Mapping vs Label"
)

# stacked bar
plt.figure(figsize=(10, 6))
ct_vs_label.plot(kind="bar", stacked=True)
plt.title("Chosen CT Mapping vs Label Distribution")
plt.xlabel("CT Mapping")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "ct_vs_label_stacked_bar.png"))
plt.close()


print("\nAll done.")