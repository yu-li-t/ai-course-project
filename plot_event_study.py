# -*- coding: utf-8 -*-
"""
按老师例图样式绘制平行趋势检验图(事件研究图)
- 横轴: 政策前后(相对年份 -8 ~ +5)
- 纵轴: 估计系数
- 每个点带 95% 置信区间误差棒
- 政策时点(-1 与 0 之间)竖虚线, y=0 水平参考线
- 黑白学术风格, 中文宋体 + 数字 Times New Roman
"""
import os, sys, io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")

for f in [r"C:\Windows\Fonts\simsun.ttc", r"C:\Windows\Fonts\times.ttf"]:
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

ev = pd.read_csv(os.path.join(OUT, "事件研究_系数表.csv"))
ev = ev.sort_values("event_time").reset_index(drop=True)
x = ev["event_time"].values
b = ev["coef"].values
lo = ev["lo"].values
hi = ev["hi"].values
base = ev["is_base"].astype(bool).values
err = np.where(base, 0, np.vstack([b - lo, hi - b]))   # 基准期不画误差棒

fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=300)

# 0 水平参考线 + 政策时点竖虚线(画在 -1 与 0 之间)
ax.axhline(0, color="black", lw=0.9, zorder=1)
ax.axvline(-0.5, color="black", lw=1.0, ls="--", zorder=1)

# 误差棒与系数点(黑白)
ax.errorbar(x[~base], b[~base], yerr=err[:, ~base], fmt="none",
            ecolor="black", elinewidth=1.0, capsize=3, capthick=0.9, zorder=2)
ax.plot(x, b, marker="o", ms=5.2, color="black", mfc="black", mec="black",
        linestyle="-", lw=1.1, zorder=3)

ax.set_xlabel("政策前后", fontsize=11)
ax.set_ylabel("估计系数", fontsize=11)
ax.set_xticks(range(-8, 6))
ax.set_xlim(-8.6, 5.6)
ymin, ymax = lo[~base].min() - 0.03, hi[~base].max() + 0.03
ax.set_ylim(ymin, ymax)
ax.tick_params(axis="both", labelsize=9.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(False)

ax.set_title("图 2  跨境电商综试区企业规模平行趋势", fontsize=12, pad=12)

fig.subplots_adjust(left=0.12, right=0.97, top=0.90, bottom=0.13)
fig.savefig(os.path.join(OUT, "图2_平行趋势检验.png"), dpi=300)
print("saved")
