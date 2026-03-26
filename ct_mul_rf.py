# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import shutil

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    accuracy_score
)

import ct_value.f1_statistic as f1
import ct_value.f3_mapping as f3


# ===============================
# 路徑
# ===============================
DATASET_DIR = "dataset/NB15"
TRAIN_PATH = os.path.join(DATASET_DIR, "train.csv")
TEST_PATH  = os.path.join(DATASET_DIR, "test.csv")

RESULT_DIR = "ct_dynamic_full"
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
# 2:1 sampling
# ===============================
def build_subset(df, attack=None):

    normal = df[df["label"] == 0]

    if attack is None:
        attack_df = df[df["label"] == 1]
    else:
        attack_df = df[df["attack_cat"] == attack]

    if len(attack_df) == 0:
        return None, None

    n_normal = min(len(normal), 2 * len(attack_df))
    normal_sample = normal.sample(n=n_normal, random_state=42)

    subset = pd.concat([normal_sample, attack_df]).sample(frac=1, random_state=42)

    if attack is None:
        y = subset["label"].values
    else:
        y = (subset["attack_cat"] == attack).astype(int).values

    return subset.reset_index(drop=True), y


# ===============================
# CT
# ===============================
def compute_ct(subset, y, name):

    save = os.path.join(FC_DIR, name)
    if os.path.exists(save):
        shutil.rmtree(save)
    os.makedirs(save)

    X = subset[get_features(subset)]
    y_series = pd.Series(np.asarray(y).reshape(-1), name="label")

    df_ct = pd.concat([X, y_series], axis=1)

    f1.statistic(df_ct, out_dir=save)

    n1 = (y_series == 1).sum()
    n0 = (y_series == 0).sum()

    ratio = n1 / max(n0, 1)
    ct, _ = f3.load_ratio_table(save, ratio, n1 >= n0, return_time=True)

    return ct


def map_ct(df, ct):
    X = sanitize(df[get_features(df)])
    ct_df, _ = f3.map_features_to_ct(X, ct, return_time=True)
    return ct_df


# ===============================
# Dynamic CT
# ===============================
def dynamic_ct_transform_full(df, ct_models, top20_map):

    valid_ct = list(top20_map.keys())  # 🔥 保證一致

    mapped = {}
    for name in valid_ct:
        mapped[name] = map_ct(df, ct_models[name])

    final_features = []
    chosen_ct = []

    for i in range(len(df)):

        best_abs = -1
        best_ct_name = None

        for name in valid_ct:

            feats = top20_map[name]
            vals = mapped[name].iloc[i][feats].values

            ct_sum = np.sum(vals)

            if abs(ct_sum) > best_abs:
                best_abs = abs(ct_sum)
                best_ct_name = name

        full_vals = mapped[best_ct_name].iloc[i].values

        final_features.append(full_vals)
        chosen_ct.append(best_ct_name)

    return pd.DataFrame(final_features), chosen_ct


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


# ===============================
# Step1：建立CT（過濾None）
# ===============================
ct_models = {}

subset, y = build_subset(train)
ct_models["global"] = compute_ct(subset, y, "global")

attacks = train.loc[train["label"] == 1, "attack_cat"].unique()

for atk in attacks:

    subset, y = build_subset(train, atk)

    if subset is None:
        print(f"Skip CT: atk_{atk}")
        continue

    name = f"atk_{atk}"
    ct_models[name] = compute_ct(subset, y, name)


print("Valid CT:", list(ct_models.keys()))


# ===============================
# Step2：top20（只保留有效CT）
# ===============================
top20_map = {}

for name, ct in ct_models.items():

    print("Selecting features:", name)

    if name == "global":
        subset, y = build_subset(train)
    else:
        atk = name.replace("atk_", "")
        try:
            atk = int(atk)
        except:
            pass

        subset, y = build_subset(train, atk)

    if subset is None:
        continue

    X_ct = map_ct(subset, ct)

    rf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    rf.fit(X_ct, y)

    imp = pd.Series(rf.feature_importances_, index=X_ct.columns)
    top20 = imp.sort_values(ascending=False).head(20).index.tolist()

    top20_map[name] = top20


print("Final CT used:", list(top20_map.keys()))


# ===============================
# Step3：Dynamic CT
# ===============================
train_ct, train_choice = dynamic_ct_transform_full(train, ct_models, top20_map)
test_ct, test_choice   = dynamic_ct_transform_full(test, ct_models, top20_map)

print("Train shape:", train_ct.shape)


# ===============================
# Step4：訓練
# ===============================
rf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(train_ct, y_train)


# ===============================
# Step5：預測
# ===============================
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

TPR = tp / (tp + fn) if (tp + fn) > 0 else 0
TNR = tn / (tn + fp) if (tn + fp) > 0 else 0
PPV = tp / (tp + fp) if (tp + fp) > 0 else 0
NPV = tn / (tn + fn) if (tn + fn) > 0 else 0

print("\n===== Final Metrics =====")
print(f"ACC   : {ACC:.4f}")
print(f"AUROC : {AUROC:.4f}")
print(f"F1    : {F1:.4f}")
print(f"TPR   : {TPR:.4f}")
print(f"TNR   : {TNR:.4f}")


# ===============================
# 存檔
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
    "chosen_ct": test_choice
}).to_csv(os.path.join(RESULT_DIR, "predictions.csv"), index=False)

print("\nSaved all results.")