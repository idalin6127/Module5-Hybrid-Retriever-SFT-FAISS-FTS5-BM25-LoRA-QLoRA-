# make_report_figs.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
ART = BASE / "eval_artifacts"

# 读取数据
scores = pd.read_csv(ART / "scores_by_method_and_k.csv")
alpha = pd.read_csv(ART / "alpha_sweep_k3.csv")

# --- 图1：Recall@k by method ---
for metric in ["Recall", "MRR", "nDCG"]:
    # 把 @1/@3/@5 三列取出来
    cols = [f"{metric}@1", f"{metric}@3", f"{metric}@5"]
    if not all(c in scores.columns for c in cols):
        continue
    df = scores[["method"] + cols].set_index("method")

    plt.figure()
    df.plot(kind="bar")  # 不指定颜色（用默认）
    plt.title(f"{metric}@k by Method (k=1/3/5)")
    plt.xlabel("Method")
    plt.ylabel(metric)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(ART / f"fig_{metric.lower()}_by_method.png", dpi=200)
    plt.close()

# --- 图2：alpha 扫描（k=3）---
for m in ["Recall@3", "MRR@3", "nDCG@3"]:
    if m not in alpha.columns:
        continue
    plt.figure()
    plt.plot(alpha["alpha"], alpha[m], marker="o")
    plt.title(f"{m} vs alpha (weighted-sum, k=3)")
    plt.xlabel("alpha")
    plt.ylabel(m)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(ART / f"fig_alpha_sweep_{m.replace('@','_')}.png", dpi=200)
    plt.close()

print("[OK] Saved figures to", ART)
