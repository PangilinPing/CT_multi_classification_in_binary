import os
import pandas as pd
import numpy as np
import time

DEFAULT_CT_VALUE = 0  # 缺值補的預設值

def load_ratio_table(count_dir, ratio, black_more, return_time: bool = False, version = True):
    """
    從 feature_count 讀每個特徵的計數表，計算 CT 映射表。
    只計「計算時間」（exclude I/O）；回傳 table 或 (table, compute_secs)。

    table 結構: { column_name: { feature_value(str): ct_value(float) } }
    """
    table = {}
    compute_secs = 0.0

    for file in os.listdir(count_dir):  # 目錄列舉視為 I/O，不計時
        if not file.endswith(".csv"):
            continue

        col_name = file[:-4]
        path = os.path.join(count_dir, file)

        # --- I/O：讀檔，不計時 ---
        map_df = pd.read_csv(path, low_memory=False)

        # --- 計算：開始計時 ---
        t0 = time.perf_counter()

        # 確保 feature_value 為字串（避免型別 mismatch）
        map_df["feature_value"] = map_df["feature_value"].astype(str)

        full_count = map_df["full_count"]
        benign_count = map_df["benign_count"]
        ratio = 1
        # 增量版
        if version:
            if black_more:
                malicious_count = map_df["full_count"] - map_df["benign_count"]
                benign_count = map_df["benign_count"] * ratio
                full_count = malicious_count + benign_count
            else:
                malicious_count = (map_df["full_count"] - map_df["benign_count"]) * ratio
                benign_count = map_df["benign_count"]
                full_count = malicious_count + benign_count
        # else:
        # # 減量版
        #     if black_more:
        #         malicious_count = (map_df["full_count"] - map_df["benign_count"])* ratio
        #         benign_count = map_df["benign_count"] 
        #         full_count = malicious_count + benign_count
        #     else:
        #         malicious_count = (map_df["full_count"] - map_df["benign_count"]) 
        #         benign_count = map_df["benign_count"]* ratio
        #         full_count = malicious_count + benign_count

        # 計算 CT 值（與你原本邏輯一致）
        # ct_value = np.log(full_count + 1) * ((benign_count / full_count) - 0.5)
        ct_value = ((benign_count / full_count) - 0.5)

        # 建立該欄位的映射 dict
        table[col_name] = dict(zip(map_df["feature_value"], ct_value))

        compute_secs += (time.perf_counter() - t0)

    return (table, compute_secs) if return_time else table


def map_features_to_ct(df: pd.DataFrame, ct_table, default_value: float = DEFAULT_CT_VALUE,
                       return_time: bool = False):
    """
    將 DataFrame 的每個欄位值映射成 CT 值。
    沒有 I/O；回傳 df_ct 或 (df_ct, compute_secs)。
    """
    t0 = time.perf_counter()
    tx = time.perf_counter()
    df_ct = df.copy()
    # print('1----------------------')
    # print(time.perf_counter() - tx)
    # print('----------------------')

    tx = time.perf_counter()
    print('2----------------------')
    for col in df_ct.columns:
        if col not in ct_table:
            # 沒對應表就跳過；你也可改成 raise 或記錄 warning
            # print(f"⚠️ {col} 沒有對應的 CT table，跳過")
            continue

        # 映射（先轉字串以對應到 table 的 key）
        df_ct[col] = df_ct[col].astype(str).map(ct_table[col]).fillna(default_value)
    print(time.perf_counter() - tx)
    print('----------------------')
    compute_secs = time.perf_counter() - t0
    return (df_ct, compute_secs) if return_time else df_ct

import hdbscan

def map_ct_with_hdbscan_models(
    df: pd.DataFrame, ct_table, hdb_models: dict, default_value: float = DEFAULT_CT_VALUE, return_time: bool = False
):

    """
    使用 column-wise HDBSCAN models 進行 mapping

    回傳：
        labels_df    : 每個欄位的 cluster label
        strength_df  : 每個欄位的 membership strength
    """

    labels = {}
    strengths = {}

    ct_df = map_features_to_ct(df, ct_table, default_value, False)
    
    for col, hdb in hdb_models.items():
        if col not in ct_df.columns:
            continue

        x = ct_df[[col]].values.reshape(-1,1)

        lbl, _ = hdbscan.approximate_predict(hdb, x)
        print(col, hdbscan.approximate_predict(hdb, np.array([0]).reshape(-1,1)))
        labels[col] = lbl
        # strengths[col] = strg

    labels_df = pd.DataFrame(labels, index=ct_df.index)
    # strength_df = pd.DataFrame(strengths, index=ct_df.index)  strength_df

    return labels_df, 0


def map_features_to_ct_fused_by_hdbscan(
    raw_train_df,
    raw_test_df,
    ct_train_df,
    ct_test_df,
    threshold=0.1
):
    """
    正確版本流程：

    raw → HDBSCAN
         ↓
    cluster
         ↓
    cluster_mean_CT (用 ct_train 計算)
         ↓
    strong? → cluster_mean_CT
    weak?   → original_CT
    """

    fused_df = ct_test_df.copy()

    total_missing = 0
    assigned_missing = 0

    for col in raw_train_df.columns:

        print("Processing:", col)

        # 1️⃣ RAW HDBSCAN
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=50,
            min_samples=20,
            prediction_data=True
        )

        hdb.fit(raw_train_df[[col]].values.reshape(-1,1))

        # 2️⃣ test raw 分群
        labels_test, _ = hdbscan.approximate_predict(
            hdb,
            raw_test_df[[col]].values.reshape(-1,1)
        )

        # 3️⃣ train raw 分群
        labels_train, _ = hdbscan.approximate_predict(
            hdb,
            raw_train_df[[col]].values.reshape(-1,1)
        )

        # 4️⃣ cluster mean CT（用 train 的 CT）
        train_cluster_df = pd.DataFrame({
            "cluster": labels_train,
            "ct": ct_train_df[col].values
        })

        cluster_means = train_cluster_df.groupby("cluster")["ct"].mean().to_dict()

        strong_clusters = {
            cid: mean for cid, mean in cluster_means.items()
            if abs(mean) > threshold
        }

        # 5️⃣ 缺值統計
        missing_mask = ct_test_df[col] == DEFAULT_CT_VALUE
        total_missing += missing_mask.sum()

        # 6️⃣ 強群替換
        for cid, mean_ct in strong_clusters.items():

            mask = (labels_test == cid)

            assigned_missing += np.sum(mask & missing_mask)

            fused_df.loc[mask, col] = mean_ct

    return fused_df, total_missing, assigned_missing


# import os
# import pandas as pd
# import numpy as np
# import time

# DEFAULT_CT_VALUE = 0  # 缺值補的預設值

# def load_ratio_table(count_dir, ratio, black_more, return_time: bool = False):
#     """
#     從 feature_count 讀每個特徵的計數表，計算 CT 映射表。
#     只計「計算時間」（exclude I/O）；回傳 table 或 (table, compute_secs)。

#     table 結構: { column_name: { feature_value(str): ct_value(float) } }
#     """
#     table = {}
#     compute_secs = 0.0

#     for file in os.listdir(count_dir):  # 目錄列舉視為 I/O，不計時
#         if not file.endswith(".csv"):
#             continue

#         col_name = file[:-4]
#         path = os.path.join(count_dir, file)

#         # --- I/O：讀檔，不計時 ---
#         map_df = pd.read_csv(path, low_memory=False)

#         # --- 計算：開始計時 ---
#         t0 = time.perf_counter()

#         # 確保 feature_value 為字串（避免型別 mismatch）
#         map_df["feature_value"] = map_df["feature_value"].astype(str)

#         pvalue = map_df['benign_count'] / map_df['full_count']
#         # 計算 CT 值（與你原本邏輯一致）
#         if black_more:
#             ct_value = pvalue /(pvalue + (1 - pvalue) *  ratio ) - 0.5
#         else:
#             ct_value = pvalue * ratio/(pvalue * ratio + (1 - pvalue)) - 0.5
#         # ct_value = ((benign_count / full_count) - 0.5)

#         # 建立該欄位的映射 dict
#         table[col_name] = dict(zip(map_df["feature_value"], ct_value))

#         compute_secs += (time.perf_counter() - t0)

#     return (table, compute_secs) if return_time else table


# def map_features_to_ct(df: pd.DataFrame, ct_table, default_value: float = DEFAULT_CT_VALUE,
#                        return_time: bool = False):
#     """
#     將 DataFrame 的每個欄位值映射成 CT 值。
#     沒有 I/O；回傳 df_ct 或 (df_ct, compute_secs)。
#     """
#     t0 = time.perf_counter()
#     tx = time.perf_counter()
#     df_ct = df.copy()
#     # print('1----------------------')
#     # print(time.perf_counter() - tx)
#     # print('----------------------')

#     tx = time.perf_counter()
#     print('2----------------------')
#     for col in df_ct.columns:
#         if col not in ct_table:
#             # 沒對應表就跳過；你也可改成 raise 或記錄 warning
#             # print(f"⚠️ {col} 沒有對應的 CT table，跳過")
#             continue

#         # 映射（先轉字串以對應到 table 的 key）
#         df_ct[col] = df_ct[col].astype(str).map(ct_table[col]).fillna(default_value)
#     print(time.perf_counter() - tx)
#     print('----------------------')
#     compute_secs = time.perf_counter() - t0
#     return (df_ct, compute_secs) if return_time else df_ct
