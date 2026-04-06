import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULT_DIR = "ct_sum_results"

# 讀取先前存好的 SHAP CSV
shap_csv_path = os.path.join(RESULT_DIR, "ctsum_shap_values.csv")
shap_df = pd.read_csv(shap_csv_path)

# 移除不是特徵的欄位
non_feature_cols = ["sample_index", "label", "pred", "score"]
feature_cols = [c for c in shap_df.columns if c not in non_feature_cols]

# 取出純 SHAP feature matrix
shap_array = shap_df[feature_cols].values

# 計算統計量
mean_abs_shap = np.abs(shap_array).mean(axis=0)
mean_shap = shap_array.mean(axis=0)
max_abs_shap = np.abs(shap_array).max(axis=0)

# 整理成表格
shap_stats_df = pd.DataFrame({
    "feature": feature_cols,
    "mean_abs_shap": mean_abs_shap,
    "mean_shap": mean_shap,
    "max_abs_shap": max_abs_shap
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

# 存檔
save_path = os.path.join(RESULT_DIR, "shap_feature_statistics.csv")
shap_stats_df.to_csv(save_path, index=False, encoding="utf-8-sig")

# 顯示前20名
topk = min(20, len(shap_stats_df))
print("Top important features by mean |SHAP|:")
print(shap_stats_df.head(topk))

# 顯示最高分特徵
top1 = shap_stats_df.iloc[0]
print("\nMost important feature:")
print(f"Feature       : {top1['feature']}")
print(f"Mean |SHAP|   : {top1['mean_abs_shap']:.6f}")
print(f"Mean SHAP     : {top1['mean_shap']:.6f}")
print(f"Max |SHAP|    : {top1['max_abs_shap']:.6f}")

# 畫圖
plot_df = shap_stats_df.head(topk).iloc[::-1]

plt.figure(figsize=(9, max(6, topk * 0.35)))
plt.barh(plot_df["feature"], plot_df["mean_abs_shap"])
plt.xlabel("Mean |SHAP value|")
plt.ylabel("Feature")
plt.title(f"Top {topk} SHAP Feature Importance")
plt.tight_layout()

plot_path = os.path.join(RESULT_DIR, "shap_feature_statistics_top20.png")
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"\nSaved CSV to: {save_path}")
print(f"Saved plot to: {plot_path}")