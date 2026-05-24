"""
scenario_output.py
==================
将 ESG 情景路径导出为标准格式（CSV / Prophet .ESC）。

输出列（CSV）：
    scenario        : 情景编号（1–N）
    time_yr         : 时间（年，从 0 开始）
    rfr_short       : 短期无风险利率（年化，连续复利）
    disc_factor     : 累计折现因子 exp(−∫r dt)
    eq_total_return : 股票累计总回报（以 1.0 为基准）
    credit_spread   : 信用利差（年化），CIR 模型输出
    fx_rate         : 汇率路径（以 1.0 为基准，Garman-Kohlhagen）
    prop_return     : 房地产累计回报指数（以 1.0 为基准，Proxy 模型）
    cpi_index       : CPI 指数路径（以 1.0 为基准）
    inflation_rate  : 瞬时通胀率（年化）

Prophet .ESC 格式（FIS Prophet 接口）：
    仅导出 rfr_short, eq_total_return, disc_factor（Prophet 标准接口列）
"""

from pathlib import Path
import numpy as np
import pandas as pd


def export_to_csv(
    r_paths: np.ndarray,
    disc_factors: np.ndarray,
    eq_tr: np.ndarray,
    dt: float,
    output_path: str = "output/esg_scenarios.csv",
    cs_paths: np.ndarray = None,
    fx_paths: np.ndarray = None,
    prop_paths: np.ndarray = None,
    cpi_paths: np.ndarray = None,
    pi_paths: np.ndarray = None,
) -> pd.DataFrame:
    """
    导出全部情景路径至 CSV。

    参数
    ----
    r_paths      : shape (n_scenarios, n_steps+1)，短期利率
    disc_factors : shape (n_scenarios, n_steps+1)，累计折现因子
    eq_tr        : shape (n_scenarios, n_steps+1)，股票总回报
    dt           : 时间步长（年）
    output_path  : 输出文件路径
    cs_paths     : 信用利差路径（可选）
    fx_paths     : 汇率路径（可选）
    prop_paths   : 房地产回报路径（可选）
    cpi_paths    : CPI 指数路径（可选）
    pi_paths     : 通胀率路径（可选）

    返回
    ----
    df : 导出的 DataFrame
    """
    n_scenarios, n_steps_plus1 = r_paths.shape
    n_steps = n_steps_plus1 - 1
    times = np.arange(n_steps_plus1) * dt

    rows = []
    for s in range(n_scenarios):
        for i in range(n_steps_plus1):
            row = {
                "scenario":        s + 1,
                "time_yr":         round(float(times[i]), 6),
                "rfr_short":       round(float(r_paths[s, i]), 8),
                "disc_factor":     round(float(disc_factors[s, i]), 8),
                "eq_total_return": round(float(eq_tr[s, i]), 8),
                "credit_spread":   round(float(cs_paths[s, i]) if cs_paths is not None else 0.0085, 8),
                "fx_rate":         round(float(fx_paths[s, i]) if fx_paths is not None else 1.0, 8),
                "prop_return":     round(float(prop_paths[s, i]) if prop_paths is not None else 1.0, 8),
                "cpi_index":       round(float(cpi_paths[s, i]) if cpi_paths is not None else 1.0, 8),
                "inflation_rate":  round(float(pi_paths[s, i]) if pi_paths is not None else 0.025, 8),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[输出] 已保存 {n_scenarios} 条情景至: {output_path}（共 {len(df):,} 行，10 列）")
    return df


def export_summary_stats(
    df: pd.DataFrame,
    output_path: str = "output/esg_summary.csv",
) -> pd.DataFrame:
    """按时间点汇总各因子的统计特征（均值、P5、P50、P95）。"""
    summary = df.groupby("time_yr").agg(
        rfr_mean=("rfr_short",       "mean"),
        rfr_p5=("rfr_short",         lambda x: np.percentile(x, 5)),
        rfr_p50=("rfr_short",        "median"),
        rfr_p95=("rfr_short",        lambda x: np.percentile(x, 95)),
        eq_mean=("eq_total_return",  "mean"),
        eq_p5=("eq_total_return",    lambda x: np.percentile(x, 5)),
        eq_p95=("eq_total_return",   lambda x: np.percentile(x, 95)),
        cs_mean=("credit_spread",    "mean"),
        cs_p5=("credit_spread",      lambda x: np.percentile(x, 5)),
        cs_p95=("credit_spread",     lambda x: np.percentile(x, 95)),
        fx_mean=("fx_rate",          "mean"),
        prop_mean=("prop_return",    "mean"),
        prop_p5=("prop_return",      lambda x: np.percentile(x, 5)),
        cpi_mean=("cpi_index",       "mean"),
        infl_mean=("inflation_rate", "mean"),
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
    导出 FIS Prophet .ESC 格式（标准接口列：rfr_short, eq_total_return, disc_factor）。
    格式：每行为一个时间步，列为空格分隔的浮点数。
    """
    n_scenarios, n_steps_plus1 = r_paths.shape

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="ascii") as f:
        f.write(f"SCENARIOS {n_scenarios}\n")
        f.write(f"TIMESTEP {dt:.8f}\n")
        f.write(f"COLUMNS rfr_short eq_total_return disc_factor\n")
        for s in range(n_scenarios):
            f.write(f"SCENARIO {s + 1}\n")
            for i in range(n_steps_plus1):
                f.write(
                    f"{r_paths[s, i]:.8f}  "
                    f"{eq_tr[s, i]:.8f}  "
                    f"{disc_factors[s, i]:.8f}\n"
                )

    print(f"[输出] Prophet .ESC 文件已保存至: {output_path}")
