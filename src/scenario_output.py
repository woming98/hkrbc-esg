"""
scenario_output.py
==================
将 ESG 情景路径导出为标准格式（CSV / Prophet .ESC 兼容格式）。

输出列说明：
- scenario    : 情景编号（1 至 n_scenarios）
- time_step   : 时间步编号（0, 1, 2, ...）
- time_yr     : 时间（年），如 0.0833 = 1/12 年
- rfr_short   : 短期无风险利率（年化，连续复利）
- disc_factor : 累计折现因子 exp(-∫₀ᵗ r du)
- eq_total_return : 股票累计总回报（以 1.0 为基准）
- credit_spread   : 信用利差（年化），简化模型直接输出
- fx_rate         : 汇率（相对于基准，1.0 = 不变）

HKRBC 用途：将此 CSV 导入 Prophet，按情景逐条计算 BEL，取均值后与确定性 BEL 相减得 TVOG。
"""

import numpy as np
import pandas as pd
from pathlib import Path


def export_to_csv(
    r_paths: np.ndarray,
    disc_factors: np.ndarray,
    eq_tr: np.ndarray,
    dt: float,
    output_path: str = "output/esg_scenarios.csv",
    cs_paths: np.ndarray = None,
    fx_paths: np.ndarray = None,
) -> pd.DataFrame:
    """
    将路径数组导出为长格式 CSV。

    参数
    ----
    r_paths      : shape (n_scenarios, n_steps+1)，短期利率路径
    disc_factors : shape (n_scenarios, n_steps+1)，累计折现因子路径
    eq_tr        : shape (n_scenarios, n_steps+1)，股票总回报路径
    dt           : 时间步长（年）
    output_path  : 输出文件路径
    cs_paths     : shape (n_scenarios, n_steps+1)，信用利差路径（可选）
    fx_paths     : shape (n_scenarios, n_steps+1)，汇率路径（可选）

    返回
    ----
    df : 长格式 DataFrame
    """
    n_scenarios, n_steps_plus1 = r_paths.shape
    n_steps = n_steps_plus1 - 1

    records = []
    for s in range(n_scenarios):
        for i in range(n_steps + 1):
            row = {
                "scenario": s + 1,
                "time_step": i,
                "time_yr": round(i * dt, 6),
                "rfr_short": round(float(r_paths[s, i]), 8),
                "disc_factor": round(float(disc_factors[s, i]), 8),
                "eq_total_return": round(float(eq_tr[s, i]), 8),
                "credit_spread": round(float(cs_paths[s, i]) if cs_paths is not None else 0.0085, 8),
                "fx_rate": round(float(fx_paths[s, i]) if fx_paths is not None else 1.0, 8),
            }
            records.append(row)

    df = pd.DataFrame(records)

    # 确保输出目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[输出] 已保存 {n_scenarios} 条情景至: {output_path}")
    print(f"       共 {len(df):,} 行，时间跨度 {n_steps * dt:.1f} 年")
    return df


def export_summary_stats(df: pd.DataFrame, output_path: str = "output/esg_summary.csv") -> pd.DataFrame:
    """
    导出各时间步的情景统计摘要（均值、标准差、百分位）。
    用于监管报告和模型验证文档。
    """
    summary = df.groupby("time_yr").agg(
        rfr_mean=("rfr_short", "mean"),
        rfr_p5=("rfr_short", lambda x: np.percentile(x, 5)),
        rfr_p50=("rfr_short", "median"),
        rfr_p95=("rfr_short", lambda x: np.percentile(x, 95)),
        eq_mean=("eq_total_return", "mean"),
        eq_p5=("eq_total_return", lambda x: np.percentile(x, 5)),
        eq_p95=("eq_total_return", lambda x: np.percentile(x, 95)),
        disc_mean=("disc_factor", "mean"),
    ).reset_index()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"[输出] 情景统计摘要已保存至: {output_path}")
    return summary


def export_prophet_esc(
    r_paths: np.ndarray,
    disc_factors: np.ndarray,
    eq_tr: np.ndarray,
    dt: float,
    output_path: str = "output/esg_scenarios.ESC",
) -> None:
    """
    导出为 Prophet .ESC 格式（ASCII 文本，FIS Prophet 可直接读取）。
    格式：每行为一个时间步，列为 rfr_short, eq_total_return, disc_factor。
    """
    n_scenarios, n_steps_plus1 = r_paths.shape
    n_steps = n_steps_plus1 - 1

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="ascii") as f:
        f.write(f"SCENARIO_COUNT  {n_scenarios}\n")
        f.write(f"TIME_STEPS      {n_steps}\n")
        f.write(f"DT_YEARS        {dt:.6f}\n")
        f.write("VARIABLES       RFR_SHORT  EQ_TOTAL_RETURN  DISC_FACTOR\n\n")

        for s in range(n_scenarios):
            f.write(f"SCENARIO  {s + 1}\n")
            for i in range(n_steps + 1):
                f.write(
                    f"  {r_paths[s, i]:.8f}  "
                    f"{eq_tr[s, i]:.8f}  "
                    f"{disc_factors[s, i]:.8f}\n"
                )
            f.write("\n")

    print(f"[输出] Prophet .ESC 文件已保存至: {output_path}")
