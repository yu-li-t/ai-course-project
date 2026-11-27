# -*- coding: utf-8 -*-
"""
事件研究法(平行趋势检验)回归: 生成各相对年份的估计系数与95%置信区间
Y = 企业规模(Size, 作为规模的代理)
模型: Size_it = 企业FE + 年份FE + Σ_k β_k·1[event_time=k] + controls + ε, 聚类到企业
基准期: event_time = -1 (省略)
"""
import os, sys, io
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import statsmodels.api as sm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")

p = os.path.join(OUT, "上市公司面板_含Treat_Cross.xlsx")
df = pd.read_excel(p)
df = df[["stkcd", "year", "Size", "Lev", "ROA", "Growth", "TobinQ", "ListAge", "Cross", "Treat"]].copy()
for c in ["stkcd", "year", "Cross", "Treat"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["stkcd", "year", "Size"]).copy()
df["stkcd"] = df["stkcd"].astype(int)
df["year"] = df["year"].astype(int)

# 每个处理组企业的首次获批年份 G_i
g = df[df["Cross"] == 1].groupby("stkcd")["year"].min().rename("G")
df = df.merge(g, on="stkcd", how="left")     # 从未获批企业 G=NaN -> 纯控制组

# 相对事件时间
df["event_time"] = df["year"] - df["G"]

# 事件窗口 -8..+5; 纯控制组 all dummy=0(与基准期-1同组)
K = list(range(-8, 6))
for k in K:
    if k == -1:
        continue
    df[f"D_{k}"] = ((df["event_time"] == k)).astype(float)
# 控制组(G缺失)所有 dummy 置 0
ctrl = df["G"].isna()
for k in K:
    if k == -1:
        continue
    df.loc[ctrl, f"D_{k}"] = 0.0

# 排除 event_time 在窗口外的处理组观测(保留纯控制组)
win = (~ctrl) & ((df["event_time"] < -8) | (df["event_time"] > 5))
df = df[~win].copy()

dummy_cols = [f"D_{k}" for k in K if k != -1]
controls = ["Lev", "ROA", "Growth", "TobinQ", "ListAge"]
# 控制变量缺失 -> 填中位数, 避免样本流失
for c in controls:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df[c] = df[c].fillna(df[c].median())

df = df.set_index(["stkcd", "year"])
y = df["Size"]
X = sm.add_constant(df[dummy_cols + controls])

mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
res = mod.fit(cov_type="clustered", cluster_entity=True)

rows = []
# 基准期-1 = 0
rows.append({"event_time": -1, "coef": 0.0, "se": 0.0, "lo": 0.0, "hi": 0.0, "p": np.nan, "is_base": True})
for k in K:
    if k == -1:
        continue
    name = f"D_{k}"
    b = res.params[name]; se = res.std_errors[name]
    rows.append({"event_time": k, "coef": b, "se": se,
                 "lo": b - 1.96 * se, "hi": b + 1.96 * se,
                 "p": res.pvalues[name], "is_base": False})
ev = pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)

summary = [
    f"样本量(公司-年度): {len(df)}",
    f"处理组公司数: {df.index.get_level_values(0).nunique() if False else int((~ctrl.reindex(df.index, fill_value=False)).sum())}",
    f"R2(within): {res.rsquared_within:.4f}",
    "",
    "event_time   coef      se       95%CI低    95%CI高    p值",
]
for _, r in ev.iterrows():
    summary.append(f"{int(r['event_time']):>5}   {r['coef']:>8.4f}  {r['se']:>7.4f}  "
                   f"{r['lo']:>8.4f}  {r['hi']:>8.4f}  {'' if r['is_base'] else format(r['p'],'.4f')}")
ev.to_csv(os.path.join(OUT, "事件研究_系数表.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(OUT, "事件研究_回归摘要.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(summary))
print("\n".join(summary))
