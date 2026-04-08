# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages


# ===============================
# 路徑設定
# ===============================

MODEL_FILES = {
    "ct_nb": "ct_nb_results/ct_nb_metrics.csv",
    "ct_rf": "ct_rf_results/ct_rf_metrics.csv",
    "ct_sum": "ct_sum_results/ct_sum_metrics.csv",
    "nb": "nb_results/nb_metrics.csv",
    "rf": "rf_results/rf_metrics.csv",
}

SAVE_DIR = "comparison_results"
os.makedirs(SAVE_DIR, exist_ok=True)

METRICS_TO_PLOT = ["Accuracy", "AUROC", "F1", "TNR", "TPR"]

PDF_PATH = os.path.join(SAVE_DIR, "all_model_comparison_plots.pdf")

# 各模型可能對應的 SHAP 圖
SHAP_IMAGE_CANDIDATES = {
    "ct_nb": [
        "ct_nb_results/shap_bar.png",
        "ct_nb_results/shap_summary.png",
    ],
    "ct_rf": [
        "ct_rf_results/shap_bar.png",
        "ct_rf_results/shap_summary.png",
    ],
    "ct_sum": [
        "ct_sum_results/ctsum_shap_bar.png",
        "ct_sum_results/ctsum_shap_summary.png",
    ],
    "nb": [
        "nb_results/shap_bar.png",
        "nb_results/shap_summary.png",
    ],
    "rf": [
        "rf_results/shap_bar.png",
        "rf_results/shap_summary.png",
    ],
}


# ===============================
# 工具函式
# ===============================

def add_image_page_to_pdf(pdf, image_path, title):
    img = mpimg.imread(image_path)

    fig = plt.figure(figsize=(8.27, 11.69))  # A4 直式
    plt.imshow(img)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ===============================
# 讀取各模型 metrics
# ===============================

all_rows = []

for model_name, file_path in MODEL_FILES.items():
    if not os.path.exists(file_path):
        print(f"[Warning] File not found: {file_path}")
        continue

    df = pd.read_csv(file_path)

    if df.empty:
        print(f"[Warning] Empty file: {file_path}")
        continue

    row = df.iloc[0].to_dict()
    row["model"] = model_name
    all_rows.append(row)

if len(all_rows) == 0:
    raise FileNotFoundError("沒有找到任何 metrics 檔案，請確認路徑是否正確。")

metrics_df = pd.DataFrame(all_rows)

# 保留 model 在第一欄
cols = ["model"] + [c for c in metrics_df.columns if c != "model"]
metrics_df = metrics_df[cols]

# 存總表
csv_save_path = os.path.join(SAVE_DIR, "all_model_metrics_comparison.csv")
metrics_df.to_csv(
    csv_save_path,
    index=False,
    encoding="utf-8-sig"
)

print("Loaded metrics:")
print(metrics_df[["model"] + [m for m in METRICS_TO_PLOT if m in metrics_df.columns]])


# ===============================
# 畫圖 + 存 PNG + 存同一份 PDF
# ===============================

with PdfPages(PDF_PATH) as pdf:
    # ---------------------------
    # 1. 先放指標比較圖
    # ---------------------------
    for metric in METRICS_TO_PLOT:
        if metric not in metrics_df.columns:
            print(f"[Warning] Metric '{metric}' not found in loaded CSVs, skip.")
            continue

        plot_df = metrics_df[["model", metric]].copy()
        plot_df = plot_df.sort_values(by=metric, ascending=False).reset_index(drop=True)

        fig = plt.figure(figsize=(8, 5))
        bars = plt.bar(plot_df["model"], plot_df[metric])

        plt.title(f"{metric} Comparison")
        plt.xlabel("Model")
        plt.ylabel(metric)

        if metric in ["Accuracy", "AUROC", "F1", "TNR", "TPR"]:
            plt.ylim(0, 1)

        for bar, value in zip(bars, plot_df[metric]):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.4f}",
                ha="center",
                va="bottom"
            )

        plt.tight_layout()

        png_save_path = os.path.join(SAVE_DIR, f"{metric}_comparison.png")
        plt.savefig(png_save_path, dpi=200, bbox_inches="tight")
        print(f"Saved PNG: {png_save_path}")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # ---------------------------
    # 2. 再放 SHAP 圖片頁
    # ---------------------------
    print("\nAdding SHAP images into PDF...")

    for model_name, image_list in SHAP_IMAGE_CANDIDATES.items():
        for image_path in image_list:
            if os.path.exists(image_path):
                title = f"{model_name} - {os.path.basename(image_path)}"
                add_image_page_to_pdf(pdf, image_path, title)
                print(f"Added to PDF: {image_path}")
            else:
                print(f"[Warning] SHAP image not found: {image_path}")

print("\nAll comparison plots saved.")
print("Saved table:", csv_save_path)
print("Saved PDF:", PDF_PATH)