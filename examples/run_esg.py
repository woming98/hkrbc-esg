"""
run_esg.py
==========
HKRBC ESG 生成器主入口，一键生成符合 Cap. 41R Rule 19 要求的情景文件。

用法
----
    python examples/run_esg.py
    python examples/run_esg.py --config config/esg_config.yaml
    python examples/run_esg.py --scenarios 2000 --seed 123

输出
----
    output/esg_scenarios.csv   : 长格式情景 CSV（可导入任意精算平台）
    output/esg_scenarios.ESC   : Prophet .ESC 格式
    output/esg_summary.csv     : 各时间步统计摘要
    martingale_test_report.txt : 鞅检验报告
"""

import sys
import argparse
from pathlib import Path

# 确保可以从根目录找到 src 模块
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml

from src.yield_curve import YieldCurve
from src.hw1f import HullWhite1F
from src.equity import EquityGBM
from src.cholesky_corr import generate_correlated_normals, default_correlation_matrix
from src.martingale_test import test_bond_martingale, test_equity_martingale, print_report
from src.scenario_output import export_to_csv, export_summary_stats, export_prophet_esc


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_correlation_matrix(corr_cfg: dict) -> np.ndarray:
    """从配置字典构建 4×4 相关性矩阵（IR, EQ, CS, FX）。"""
    ir_eq = corr_cfg.get("ir_eq", -0.20)
    ir_cs = corr_cfg.get("ir_cs", -0.30)
    ir_fx = corr_cfg.get("ir_fx",  0.10)
    eq_cs = corr_cfg.get("eq_cs",  0.40)
    eq_fx = corr_cfg.get("eq_fx", -0.15)
    cs_fx = corr_cfg.get("cs_fx", -0.10)

    return np.array([
        [1.00, ir_eq, ir_cs, ir_fx],
        [ir_eq, 1.00, eq_cs, eq_fx],
        [ir_cs, eq_cs, 1.00, cs_fx],
        [ir_fx, eq_fx, cs_fx, 1.00],
    ])


def main(args):
    print("=" * 60)
    print("  HKRBC ESG Generator v1.0")
    print("  Cap. 41R Rule 19 — Market-Consistent Scenarios")
    print("=" * 60)

    # ── 1. 加载配置 ──────────────────────────────────────────
    cfg = load_config(args.config)
    sim = cfg["simulation"]

    n_scenarios = args.scenarios or sim["n_scenarios"]
    n_years = sim["projection_years"]
    seed = args.seed or sim["random_seed"]
    dt_map = {"monthly": 1/12, "quarterly": 0.25, "annual": 1.0}
    dt = dt_map[sim["time_step"]]
    n_steps = int(n_years / dt)

    print(f"\n[配置] 情景数={n_scenarios} | 期限={n_years}年 | 步长={sim['time_step']} | 种子={seed}")

    # ── 2. 构建 IA Schedule 4 利率曲线 ──────────────────────
    ia_cfg = cfg["ia_schedule4"]
    maturities = np.array(ia_cfg["maturities"], dtype=float)
    spot_rates = np.array(ia_cfg["spot_rates"], dtype=float)
    yc = YieldCurve(maturities, spot_rates)
    print(f"[利率曲线] 已加载 {len(maturities)} 个期限点，UFR={ia_cfg['ufr']:.1%}")

    # ── 3. 生成相关性随机数（对偶变量法，antithetic variates）────
    # 对偶变量法：前半 n/2 条情景用 Z，后半用 -Z（所有因子同步镜像）
    # 这样可将 martingale 估计方差降低约 50%，使 1,000 条情景接近 2,000 条的精度
    corr_matrix = build_correlation_matrix(cfg.get("correlation", {}))
    half = n_scenarios // 2
    Z_half = generate_correlated_normals(corr_matrix, half, n_steps, seed=seed)
    # 镜像：所有因子同时取负（保持因子间相关性结构不变）
    Z_full = np.concatenate([Z_half, -Z_half], axis=1)  # shape (n_factors, n_scenarios, n_steps)
    Z_ir, Z_eq, Z_cs, Z_fx = Z_full[0], Z_full[1], Z_full[2], Z_full[3]

    # ── 4. Hull-White 1F 利率模拟 ────────────────────────────
    hw_cfg = cfg["hull_white_1f"]
    hw = HullWhite1F(a=hw_cfg["a"], sigma=hw_cfg["sigma"], yc=yc)
    print(f"[HW1F] a={hw_cfg['a']}, σ={hw_cfg['sigma']} | 正在生成 {n_scenarios} 条利率路径...")
    # 将相关随机数 Z_ir 传入，与股票/信用等因子保持正确相关性结构
    # 同时启用对偶变量法（antithetic variates）大幅提升 martingale test 通过率
    r_paths, disc_factors = hw.simulate(
        n_scenarios, n_steps, dt, seed=seed,
        Z_external=Z_ir,
        antithetic=False,  # 已通过外部 Cholesky 控制相关性，不再额外做对偶
    )
    print(f"       利率路径生成完成。r(0)={r_paths[0,0]:.4f}")

    # ── 5. GBM 股票总回报模拟 ────────────────────────────────
    eq_cfg = cfg["equity"]
    equity = EquityGBM(sigma_eq=eq_cfg["sigma_eq"], s0=eq_cfg["s0"])
    print(f"[Equity] σ_eq={eq_cfg['sigma_eq']:.0%} | 正在生成股票路径...")
    _, eq_tr = equity.simulate(r_paths, dt, corr_Z=Z_eq)
    print(f"         股票路径生成完成。")

    # ── 6. 信用利差模拟（OU 过程，简化）─────────────────────
    cs_cfg = cfg["credit_spread"]
    cs_paths = np.zeros((n_scenarios, n_steps + 1))
    cs_paths[:, 0] = cs_cfg["cs0"]
    kappa, mean_cs, sigma_cs = cs_cfg["kappa_cs"], cs_cfg["mean_cs"], cs_cfg["sigma_cs"]
    for i in range(n_steps):
        cs_paths[:, i+1] = (
            cs_paths[:, i] + kappa * (mean_cs - cs_paths[:, i]) * dt
            + sigma_cs * np.sqrt(dt) * Z_cs[:, i]
        )
    cs_paths = np.clip(cs_paths, 0, None)  # 信用利差不能为负

    # ── 7. Martingale Test ───────────────────────────────────
    mt_cfg = cfg["martingale_test"]
    print("\n[Martingale Test] 正在验证情景的市场一致性...")
    bond_result = test_bond_martingale(
        disc_factors, yc, dt,
        check_tenors=mt_cfg["check_tenors"],
        tolerance=mt_cfg["tolerance"]
    )
    eq_result = test_equity_martingale(
        disc_factors, eq_tr, dt=dt,
        tolerance=mt_cfg["tolerance"],
        max_check_tenor=max(mt_cfg["check_tenors"]),
    )
    print_report(bond_result, eq_result)

    if not (bond_result["all_passed"] and eq_result["all_passed"]):
        print("\n⚠️  建议：调整 HW1F 参数（a, σ）或增加情景数量后重试。")
        print("     参考方向：误差偏大通常因 n_scenarios 过少或参数校准不足。")

    # ── 8. 导出情景文件 ──────────────────────────────────────
    out_cfg = cfg["output"]
    print(f"\n[输出] 正在写入情景文件...")

    if out_cfg.get("save_csv", True):
        df = export_to_csv(
            r_paths, disc_factors, eq_tr, dt,
            output_path=out_cfg["csv_path"],
            cs_paths=cs_paths
        )
        if out_cfg.get("save_summary", True):
            export_summary_stats(df, output_path=out_cfg["summary_path"])

    if out_cfg.get("save_esc", True):
        export_prophet_esc(r_paths, disc_factors, eq_tr, dt, output_path=out_cfg["esc_path"])

    print("\n✅ ESG 生成完成。请检查 output/ 目录下的情景文件。")
    print("   下一步：将 esg_scenarios.csv 或 .ESC 文件导入 Prophet，")
    print("           按情景计算 BEL，TVOG = 随机均值 BEL − 确定性 BEL")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HKRBC ESG Generator")
    parser.add_argument("--config", default="config/esg_config.yaml", help="配置文件路径")
    parser.add_argument("--scenarios", type=int, default=None, help="情景数量（覆盖配置文件）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（覆盖配置文件）")
    args = parser.parse_args()
    main(args)
