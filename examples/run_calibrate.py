"""
run_calibrate.py
================
G2++ 参数校准入口脚本。

用法：
    python examples/run_calibrate.py
    python examples/run_calibrate.py --csv data/swaption_vols_hkd.csv
    python examples/run_calibrate.py --csv data/my_bloomberg_vols.csv --verbose

校准完成后自动更新 config/esg_config.yaml 中的 hull_white_2f 参数。

数据文件格式（data/swaption_vols_hkd.csv）：
    expiry_yr  : option expiry（年）
    tenor_yr   : swap tenor（年）
    normal_vol_bps : ATM Bachelier implied vol（bps）
    weight     : 校准权重（可选，默认 1.0）

替换数据源：
    将 Bloomberg SWPN ATM normal vol 导出为上述格式的 CSV，
    再运行本脚本即可完成实时市场数据校准。
"""

import argparse
import os
import sys
import yaml

# 允许从项目根目录运行
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.calibrate import calibrate_G2, load_vol_surface
from src.yield_curve import YieldCurve


def build_yield_curve_from_config(cfg: dict) -> YieldCurve:
    """从 esg_config.yaml 的 ia_schedule4 字段构建 YieldCurve 对象。"""
    sch4 = cfg["ia_schedule4"]
    maturities = sch4["maturities"]
    spot_rates  = sch4["spot_rates"]
    ufr         = sch4.get("ufr", 0.038)
    llp         = sch4.get("llp", 15)
    return YieldCurve(
        maturities=maturities,
        spot_rates=spot_rates,
        ufr=ufr,
        llp=llp,
    )


def update_config(config_path: str, calib_result: dict) -> None:
    """将校准结果写回 esg_config.yaml 的 hull_white_2f 字段。"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    hw2f = cfg.setdefault("hull_white_2f", {})
    hw2f["a"]      = calib_result["a"]
    hw2f["b"]      = calib_result["b"]
    hw2f["sigma1"] = calib_result["sigma1"]
    hw2f["sigma2"] = calib_result["sigma2"]
    hw2f["rho_xy"] = calib_result["rho"]
    # 保留 term_premium（不覆盖）
    hw2f.setdefault("term_premium_Q", 0.000)
    hw2f.setdefault("term_premium_P", 0.008)

    # 添加校准元数据注释（YAML 不支持内联注释，写入 calibration_meta 字段）
    cfg["calibration_meta"] = {
        "rmse_bps":   calib_result["rmse_bps"],
        "n_points":   len(calib_result["expiries"]),
        "converged":  calib_result["converged"],
    }

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"\n✅ 校准参数已写入：{config_path}")


def main():
    parser = argparse.ArgumentParser(description="G2++ swaption vol surface 校准")
    parser.add_argument(
        "--csv",
        default="data/swaption_vols_hkd.csv",
        help="swaption vol surface CSV 路径（默认：data/swaption_vols_hkd.csv）",
    )
    parser.add_argument(
        "--config",
        default="config/esg_config.yaml",
        help="配置文件路径（默认：config/esg_config.yaml）",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=400,
        help="differential_evolution 最大迭代次数（默认：400）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认：42）",
    )
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="校准完成后不更新 esg_config.yaml",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="打印详细输出（默认开启）",
    )
    args = parser.parse_args()

    # ── 1. 加载配置 & 构建 yield curve ────────────────────────────────────
    config_path = os.path.join(os.path.dirname(__file__), "..", args.config)
    config_path = os.path.normpath(config_path)

    if not os.path.exists(config_path):
        print(f"[ERROR] 找不到配置文件：{config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    yc = build_yield_curve_from_config(cfg)
    print(f"✔ Yield curve 已加载（UFR={cfg['ia_schedule4']['ufr']*100:.1f}%，LLP={cfg['ia_schedule4']['llp']}Y）")

    # ── 2. 加载 swaption vol surface ──────────────────────────────────────
    csv_path = os.path.join(os.path.dirname(__file__), "..", args.csv)
    csv_path = os.path.normpath(csv_path)

    if not os.path.exists(csv_path):
        print(f"[ERROR] 找不到 vol surface 文件：{csv_path}")
        print("请先准备 data/swaption_vols_hkd.csv 或通过 --csv 指定路径。")
        sys.exit(1)

    vol_surface = load_vol_surface(csv_path)
    print(f"✔ Vol surface 已加载：{len(vol_surface)} 个校准点（来自 {csv_path}）")

    # ── 3. 校准 ───────────────────────────────────────────────────────────
    result = calibrate_G2(
        vol_surface=vol_surface,
        yc=yc,
        seed=args.seed,
        maxiter=args.maxiter,
        verbose=args.verbose,
    )

    # ── 4. 更新配置 ───────────────────────────────────────────────────────
    if not args.no_update:
        update_config(config_path, result)
    else:
        print("\n[--no-update] 跳过写入 esg_config.yaml")
        print(f"校准结果：a={result['a']}, b={result['b']}, "
              f"σ₁={result['sigma1']}, σ₂={result['sigma2']}, ρ={result['rho']}")

    print("\n运行 ESG 模拟（使用校准后的参数）：")
    print("  python examples/run_esg.py --ir-model HW2F")


if __name__ == "__main__":
    main()
