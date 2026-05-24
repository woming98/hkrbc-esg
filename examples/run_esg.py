"""
run_esg.py
==========
HKRBC ESG Generator v2.0 主程序。

支持：
  - IR 模型：HW1F / HW2F(G2++) / LMM（通过 ir_model 配置）
  - 模式：Q-measure（TVOG）/ P-measure（ALM/ORSA）（通过 mode 配置）
  - 六类风险因子：IR, EQ, CS(CIR), FX(GK), Property(Proxy), Inflation(OU)

用法：
    python examples/run_esg.py
    python examples/run_esg.py --config config/esg_config.yaml
    python examples/run_esg.py --ir-model HW2F --mode P --scenarios 500
"""

import argparse
import numpy as np
import yaml

from src.yield_curve import YieldCurve
from src.hw1f import HullWhite1F
from src.hw2f import HullWhite2F
from src.lmm import LMM
from src.equity import EquityGBM
from src.cir import CIRProcess
from src.fx import GarmanKohlhagen
from src.property_return import PropertyReturn
from src.inflation import InflationOU
from src.cholesky_corr import generate_correlated_normals
from src.martingale_test import test_bond_martingale, test_equity_martingale, print_report
from src.scenario_output import export_to_csv, export_summary_stats, export_prophet_esc


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(args):
    print("=" * 65)
    print("  HKRBC ESG Generator v2.0")
    print("  IR: HW1F / HW2F(G2++) / LMM")
    print("  因子: IR · EQ · CS(CIR) · FX(GK) · Property · Inflation")
    print("=" * 65)

    # ── 1. 加载配置 ────────────────────────────────────────────────
    cfg = load_config(args.config)
    sim = cfg["simulation"]
    mode = (args.mode or cfg.get("mode", "Q")).upper()
    ir_model = (args.ir_model or cfg.get("ir_model", "HW1F")).upper()

    n_scenarios = args.scenarios or sim["n_scenarios"]
    n_years = sim["projection_years"]
    seed = args.seed or sim["random_seed"]
    dt_map = {"monthly": 1 / 12, "quarterly": 0.25, "annual": 1.0}
    dt = dt_map[sim["time_step"]]
    n_steps = int(n_years / dt)

    print(f"\n[配置]")
    print(f"  模式     : {'Q-measure（风险中性 → HKRBC TVOG）' if mode=='Q' else 'P-measure（真实世界 → ALM/ORSA）'}")
    print(f"  IR 模型  : {ir_model}")
    print(f"  情景数   : {n_scenarios} | 期限: {n_years}年 | 步长: {sim['time_step']} | 种子: {seed}")

    # ── 2. 构建 IA Schedule 4 利率曲线 ────────────────────────────
    ia_cfg = cfg["ia_schedule4"]
    yc = YieldCurve(
        maturities=np.array(ia_cfg["maturities"]),
        spot_rates=np.array(ia_cfg["spot_rates"]),
    )
    print(f"\n[利率曲线] 已加载 {len(ia_cfg['maturities'])} 个期限点，UFR={ia_cfg.get('ufr',0.038):.1%}")

    # ── 3. 生成相关随机数（4 因子：IR, EQ, CS, FX）───────────────
    corr_cfg = cfg["correlation"]
    ir_eq = corr_cfg.get("ir_eq", -0.20)
    ir_cs = corr_cfg.get("ir_cs", -0.30)
    ir_fx = corr_cfg.get("ir_fx",  0.10)
    eq_cs = corr_cfg.get("eq_cs",  0.40)
    eq_fx = corr_cfg.get("eq_fx", -0.15)
    cs_fx = corr_cfg.get("cs_fx", -0.10)

    corr_matrix = np.array([
        [1.00, ir_eq, ir_cs, ir_fx],
        [ir_eq, 1.00, eq_cs, eq_fx],
        [ir_cs, eq_cs, 1.00, cs_fx],
        [ir_fx, eq_fx, cs_fx, 1.00],
    ])

    # 对偶变量法（antithetic variates）：前半段 Z，后半段 -Z
    half = n_scenarios // 2
    Z_half = generate_correlated_normals(corr_matrix, half, n_steps, seed=seed)
    Z_full = np.concatenate([Z_half, -Z_half], axis=1)
    Z_ir, Z_eq, Z_cs, Z_fx = Z_full[0], Z_full[1], Z_full[2], Z_full[3]

    print(f"\n[随机数] 4 因子 Cholesky 相关矩阵生成完成（对偶变量法）")

    # ── 4. IR 模型模拟 ─────────────────────────────────────────────
    print(f"\n[IR 模型: {ir_model}]")

    if ir_model == "HW1F":
        hw1f_cfg = cfg["hull_white_1f"]
        tp = hw1f_cfg.get(f"term_premium_{mode}", 0.0)
        hw = HullWhite1F(a=hw1f_cfg["a"], sigma=hw1f_cfg["sigma"], yc=yc, term_premium=tp)
        print(f"  a={hw1f_cfg['a']}, σ={hw1f_cfg['sigma']}, term_premium={tp:.4f}")
        r_paths, disc_factors = hw.simulate(n_scenarios, n_steps, dt, seed=seed, Z_external=Z_ir, antithetic=False)

    elif ir_model == "HW2F":
        hw2f_cfg = cfg["hull_white_2f"]
        tp = hw2f_cfg.get(f"term_premium_{mode}", 0.0)
        hw2 = HullWhite2F(
            a=hw2f_cfg["a"], b=hw2f_cfg["b"],
            sigma1=hw2f_cfg["sigma1"], sigma2=hw2f_cfg["sigma2"],
            rho_xy=hw2f_cfg["rho_xy"], yc=yc, term_premium=tp,
        )
        print(f"  a={hw2f_cfg['a']}, b={hw2f_cfg['b']}, σ₁={hw2f_cfg['sigma1']}, σ₂={hw2f_cfg['sigma2']}, ρ_xy={hw2f_cfg['rho_xy']}, term_premium={tp:.4f}")
        r_paths, disc_factors, _, _ = hw2.simulate(n_scenarios, n_steps, dt, seed=seed, Z_external=Z_ir)

    elif ir_model == "LMM":
        lmm_cfg = cfg["lmm"]
        lmm = LMM(
            tenors=lmm_cfg["tenors"],
            sigma_vols=lmm_cfg["sigma_vols"],
            rho_decay=lmm_cfg["rho_decay"],
            yc=yc,
        )
        print(f"  N={len(lmm_cfg['tenors'])} 个 tenor，σ={lmm_cfg['sigma_vols']}, ρ_decay={lmm_cfg['rho_decay']}")
        print(f"  注意：LMM 步长固定为 1 年，将覆写 projection_years={n_years}Y")
        _, r_paths_lmm, disc_paths_lmm = lmm.simulate(n_scenarios, seed=seed)
        # LMM 输出为年步长，重采样至月步长（线性插值，简化处理）
        n_lmm_steps = r_paths_lmm.shape[1] - 1
        t_lmm = np.linspace(0, n_lmm_steps, n_lmm_steps + 1)
        t_monthly = np.linspace(0, n_lmm_steps, n_steps + 1)
        r_paths = np.array([np.interp(t_monthly, t_lmm, r_paths_lmm[s]) for s in range(n_scenarios)])
        disc_factors = np.array([np.interp(t_monthly, t_lmm, disc_paths_lmm[s]) for s in range(n_scenarios)])
    else:
        raise ValueError(f"未知 ir_model: {ir_model}，请选择 HW1F / HW2F / LMM")

    print(f"  ✓ 利率路径生成完成。r(0)={r_paths[0,0]:.4f}（{r_paths[0,0]:.2%}）")

    # ── 5. 股票 GBM ────────────────────────────────────────────────
    eq_cfg = cfg["equity"]
    erp = eq_cfg.get(f"equity_risk_premium_{mode}", 0.0)
    equity = EquityGBM(sigma_eq=eq_cfg["sigma_eq"], s0=eq_cfg["s0"], equity_risk_premium=erp)
    print(f"\n[股票 GBM] σ_eq={eq_cfg['sigma_eq']:.0%}, ERP={erp:.1%}")
    _, eq_tr = equity.simulate(r_paths, dt, corr_Z=Z_eq)
    s_paths = eq_tr * eq_cfg["s0"]
    print(f"  ✓ 股票路径生成完成。期望年化回报 ≈ {r_paths[0,0] + erp:.2%}")

    # ── 6. 信用利差（CIR）─────────────────────────────────────────
    cs_cfg = cfg["credit_spread"]
    cir = CIRProcess(
        kappa=cs_cfg["kappa"],
        mu=cs_cfg["mu"],
        sigma=cs_cfg["sigma"],
        cs0=cs_cfg["cs0"],
    )
    print(f"\n[信用利差 CIR] cs0={cs_cfg['cs0']:.0%}, μ={cs_cfg['mu']:.0%}, κ={cs_cfg['kappa']}, σ={cs_cfg['sigma']}")
    cs_paths = cir.simulate(n_scenarios, n_steps, dt, corr_Z=Z_cs)
    print(f"  ✓ 信用利差路径生成完成。平均 10Y: {cs_paths[:, int(10/dt)].mean():.4f}")

    # ── 7. 汇率（Garman-Kohlhagen）────────────────────────────────
    fx_enabled = cfg.get("fx", {}).get("enabled", True)
    fx_paths = None
    if fx_enabled:
        fx_cfg = cfg["fx"]
        gk = GarmanKohlhagen(sigma_fx=fx_cfg["sigma_fx"], r_f=fx_cfg["r_f"], s0=fx_cfg["s0"])
        print(f"\n[FX GK] σ_fx={fx_cfg['sigma_fx']:.1%}, r_f={fx_cfg['r_f']:.1%}")
        fx_paths = gk.simulate(r_paths, dt, corr_Z=Z_fx)
        print(f"  ✓ 汇率路径生成完成。30Y 均值: {fx_paths[:, -1].mean():.4f}")

    # ── 8. 房地产（Proxy to Equity）──────────────────────────────
    prop_enabled = cfg.get("property_return", {}).get("enabled", True)
    prop_paths = None
    if prop_enabled:
        pr_cfg = cfg["property_return"]
        pr = PropertyReturn(
            beta=pr_cfg["beta"],
            lag_steps=pr_cfg["lag_steps"],
            sigma_idio=pr_cfg["sigma_idio"],
            prop0=pr_cfg["prop0"],
        )
        print(f"\n[房地产 Proxy] beta={pr_cfg['beta']}, lag={pr_cfg['lag_steps']}步, σ_idio={pr_cfg['sigma_idio']:.0%}")
        # 计算股票逐步对数回报
        eq_log_ret = np.log(s_paths[:, 1:] / s_paths[:, :-1])
        prop_paths = pr.simulate(eq_log_ret, dt, seed=seed + 200)
        print(f"  ✓ 房地产路径生成完成。30Y 均值: {prop_paths[:, -1].mean():.4f}")

    # ── 9. 通货膨胀（OU）─────────────────────────────────────────
    infl_enabled = cfg.get("inflation", {}).get("enabled", True)
    pi_paths, cpi_paths = None, None
    if infl_enabled:
        inf_cfg = cfg["inflation"]
        infl = InflationOU(
            kappa_pi=inf_cfg["kappa_pi"],
            mu_pi=inf_cfg["mu_pi"],
            sigma_pi=inf_cfg["sigma_pi"],
            pi0=inf_cfg["pi0"],
        )
        print(f"\n[通胀 OU] π(0)={inf_cfg['pi0']:.1%}, μ={inf_cfg['mu_pi']:.1%}, κ={inf_cfg['kappa_pi']}, σ={inf_cfg['sigma_pi']:.1%}")
        pi_paths, cpi_paths = infl.simulate(n_scenarios, n_steps, dt, seed=seed + 300)
        print(f"  ✓ 通胀路径生成完成。30Y 累计 CPI 均值: {cpi_paths[:, -1].mean():.4f}（年化 {(cpi_paths[:, -1].mean()**(1/n_years)-1):.2%}）")

    # ── 10. Martingale Test（仅 Q-mode，且 IR 非 LMM）────────────
    mt_cfg = cfg["martingale_test"]
    print()
    if mode == "Q" and ir_model != "LMM":
        print("[Martingale Test] Q-mode：验证情景市场一致性（Rule 19）...")
        bond_result = test_bond_martingale(disc_factors, yc, dt,
                                           check_tenors=mt_cfg["check_tenors"],
                                           tolerance=mt_cfg["tolerance"])
        eq_result = test_equity_martingale(disc_factors, eq_tr, dt=dt,
                                           tolerance=mt_cfg["tolerance"],
                                           max_check_tenor=max(mt_cfg["check_tenors"]))
        print_report(bond_result, eq_result)
        if not (bond_result["all_passed"] and eq_result["all_passed"]):
            print("⚠️  建议：调整参数（a, σ）或增加情景数量后重试。")
    elif mode == "Q" and ir_model == "LMM":
        print("[Martingale Test] LMM：forward rate 结构内生保证近似鞅性质，跳过外部检验。")
    else:
        print("[Martingale Test] P-mode：跳过（Real-World 期望值 ≠ 市场价格，正常现象）。")
        print("  P-mode 验证：比较情景分布与历史数据（均值、波动率、分位数）。")

    # ── 11. 导出情景文件 ──────────────────────────────────────────
    out_cfg = cfg["output"]
    print(f"\n[输出] 正在写入情景文件...")

    if out_cfg.get("save_csv", True):
        df = export_to_csv(
            r_paths, disc_factors, eq_tr, dt,
            output_path=out_cfg["csv_path"],
            cs_paths=cs_paths,
            fx_paths=fx_paths,
            prop_paths=prop_paths,
            cpi_paths=cpi_paths,
            pi_paths=pi_paths,
        )
        if out_cfg.get("save_summary", True):
            export_summary_stats(df, output_path=out_cfg["summary_path"])

    if out_cfg.get("save_esc", True):
        export_prophet_esc(r_paths, disc_factors, eq_tr, dt, output_path=out_cfg["esc_path"])

    print("\n✅ ESG 生成完成！")
    print("   输出文件（output/ 目录）：")
    print("     esg_scenarios.csv  → 10 列全因子情景（IR/EQ/CS/FX/PR/INF）")
    print("     esg_scenarios.ESC  → Prophet .ESC 格式（rfr_short/eq_TR/disc）")
    print("     esg_summary.csv    → 各时间点统计摘要（均值/P5/P50/P95）")
    print("\n   下一步：")
    print("     1. 将 esg_scenarios.csv 或 .ESC 导入 Prophet")
    print("     2. 对每条情景计算 BEL（含 GSV floor + DPB）")
    print("     3. TVOG = mean(stochastic BEL) − deterministic BEL")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HKRBC ESG Generator v2.0")
    parser.add_argument("--config",    default="config/esg_config.yaml")
    parser.add_argument("--scenarios", type=int, default=None)
    parser.add_argument("--seed",      type=int, default=None)
    parser.add_argument("--ir-model",  choices=["HW1F", "HW2F", "LMM"], default=None,
                        help="覆盖 config 中的 ir_model")
    parser.add_argument("--mode",      choices=["Q", "P"], default=None,
                        help="覆盖 config 中的 mode（Q=风险中性，P=真实世界）")
    args = parser.parse_args()
    main(args)
