# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ===============================
# 路徑設定
# ===============================

# 沒有 FS 的結果
MODEL_FILES_NO_FS = {
    "ct_nb": "ct_nb_results/ct_nb_metrics.csv",
    "ct_rf": "ct_rf_results/ct_rf_metrics.csv",
    "ct_sum": "ct_sum_results/ct_sum_metrics.csv",
    "nb": "nb_results/nb_metrics.csv",
    "rf": "rf_results/rf_metrics.csv",
}

# 有 FS 的結果
MODEL_FILES_FS = {
    "ct_nb": "feature_selection_from_shap_results/ct_nb/ct_nb_metrics.csv",
    "ct_rf": "feature_selection_from_shap_results/ct_rf/ct_rf_metrics.csv",
    "ct_sum": "feature_selection_from_shap_results/ct_sum/ct_sum_metrics.csv",
    "nb": "feature_selection_from_shap_results/nb/nb_metrics.csv",
    "rf": "feature_selection_from_shap_results/rf/rf_metrics.csv",
}

SAVE_DIR = "comparison_fs_vs_no_fs"
os.makedirs(SAVE_DIR, exist_ok=True)

METRICS_TO_PLOT = ["Accuracy", "AUROC", "F1", "TNR", "TPR"]
PDF_PATH = os.path.join(SAVE_DIR, "fs_vs_no_fs_comparison.pdf")


# ===============================
# 工具函式
# ===============================

def load_metrics(model_files, fs_label):
    rows = []

    for model_name, file_path in model_files.items():
        if not os.path.exists(file_path):
            print(f"[Warning] File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)

        if df.empty:
            print(f"[Warning] Empty file: {file_path}")
            continue

        row = df.iloc[0].to_dict()
        row["model"] = model_name
        row["fs"] = fs_label
        rows.append(row)

    return rows


# ===============================
# 讀取資料
# ===============================

all_rows = []
all_rows.extend(load_metrics(MODEL_FILES_NO_FS, "No_FS"))
all_rows.extend(load_metrics(MODEL_FILES_FS, "FS"))

if len(all_rows) == 0:
    raise FileNotFoundError("沒有找到任何 metrics 檔案，請確認路徑是否正確。")

metrics_df = pd.DataFrame(all_rows)

# 欄位順序整理
base_cols = ["model", "fs"]
other_cols = [c for c in metrics_df.columns if c not in base_cols]
metrics_df = metrics_df[base_cols + other_cols]

# 存總表
csv_save_path = os.path.join(SAVE_DIR, "fs_vs_no_fs_metrics.csv")
metrics_df.to_csv(csv_save_path, index=False, encoding="utf-8-sig")

print("Loaded metrics:")
show_cols = ["model", "fs"] + [m for m in METRICS_TO_PLOT if m in metrics_df.columns]
print(metrics_df[show_cols])


# ===============================
# 畫圖 + PDF
# ===============================

with PdfPages(PDF_PATH) as pdf:
    for metric in METRICS_TO_PLOT:
        if metric not in metrics_df.columns:
            print(f"[Warning] Metric '{metric}' not found, skip.")
            continue

        # 依模型排列，方便每個模型看到 FS / No_FS
        plot_df = metrics_df[["model", "fs", metric]].copy()

        model_order = ["ct_nb", "ct_rf", "ct_sum", "nb", "rf"]
        plot_df["model"] = pd.Categorical(plot_df["model"], categories=model_order, ordered=True)
        plot_df["fs"] = pd.Categorical(plot_df["fs"], categories=["No_FS", "FS"], ordered=True)
        plot_df = plot_df.sort_values(["model", "fs"]).reset_index(drop=True)

        x_labels = []
        values = []
        colors = []

        for _, row in plot_df.iterrows():
            x_labels.append(f"{row['model']}\n{row['fs']}")
            values.append(row[metric])

            if row["fs"] == "FS":
                colors.append("red")
            else:
                colors.append("steelblue")

        fig = plt.figure(figsize=(12, 6))
        bars = plt.bar(x_labels, values, color=colors)

        plt.title(f"{metric} Comparison (FS vs No_FS)")
        plt.xlabel("Model")
        plt.ylabel(metric)

        if metric in ["Accuracy", "AUROC", "F1", "TNR", "TPR"]:
            plt.ylim(0, 1)

        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

        plt.tight_layout()

        png_save_path = os.path.join(SAVE_DIR, f"{metric}_fs_vs_no_fs.png")
        plt.savefig(png_save_path, dpi=200, bbox_inches="tight")
        print(f"Saved PNG: {png_save_path}")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

print("\nAll comparison plots saved.")
print("Saved table:", csv_save_path)
print("Saved PDF:", PDF_PATH)