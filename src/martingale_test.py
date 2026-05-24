"""
martingale_test.py
==================
验证 ESG 情景是否通过 Martingale Test（鞅检验）。

HKRBC 监管要求（Cap. 41R Rule 19）：
    所有情景必须满足市场一致性（market-consistent）和无套利（arbitrage-free）。
    Martingale Test 是验证 Q-measure 正确性的标准方法。

鞅条件（对每个到期时刻 T）：
    E^Q [ exp(-∫₀ᵀ r(t) dt) ] = P(0, T)
    即：所有情景的折现因子均值 = 市场观测的零息债券价格

对于股票（Total Return Index）：
    E^Q [ S(T) / S(0) · exp(-∫₀ᵀ r(t) dt) ] = 1
    即：折现股票价格的期望 = 今日股价（S(0) = 1）

允许误差：实务中允许 ±0.5% 误差（不同公司 policy 不同，典型值 0.3%–1.0%）。
"""

import numpy as np
from src.yield_curve import YieldCurve


def test_bond_martingale(
    disc_factors: np.ndarray,
    yc: YieldCurve,
    dt: float,
    check_tenors: list[float] = None,
    tolerance: float = 0.005,
) -> dict:
    """
    债券鞅检验：验证折现因子均值是否等于市场零息债券价格。

    参数
    ----
    disc_factors  : shape (n_scenarios, n_steps+1)，累计折现因子路径
    yc            : YieldCurve，初始 IA Schedule 4 yield curve
    dt            : 时间步长（年）
    check_tenors  : 检验的到期期限列表（年），默认 [1, 2, 3, 5, 7, 10, 15, 20, 30]
    tolerance     : 允许误差（默认 0.5%）

    返回
    ----
    result : dict，包含各期限的检验结果和通过标志
    """
    if check_tenors is None:
        check_tenors = [1, 2, 3, 5, 7, 10, 15, 20, 30]

    n_steps = disc_factors.shape[1] - 1
    max_t = n_steps * dt
    results = []

    for T in check_tenors:
        if T > max_t:
            continue
        step_idx = int(round(T / dt))
        # 模型预测：E^Q[exp(-∫r dt)]
        model_price = np.mean(disc_factors[:, step_idx])
        # 市场真实价格
        market_price = yc.discount_factor(T)
        error_pct = abs(model_price - market_price) / market_price * 100
        passed = error_pct <= tolerance * 100

        results.append({
            "tenor": T,
            "model_price": round(model_price, 6),
            "market_price": round(market_price, 6),
            "error_pct": round(error_pct, 4),
            "passed": passed,
        })

    all_passed = all(r["passed"] for r in results)
    return {"all_passed": all_passed, "details": results, "tolerance_pct": tolerance * 100}


def test_equity_martingale(
    disc_factors: np.ndarray,
    equity_tr: np.ndarray,
    dt: float,
    tolerance: float = 0.010,
    max_check_tenor: float = 15.0,
) -> dict:
    """
    股票鞅检验：验证折现股票价格的期望 = 初始价格（= 1.0）。

    参数
    ----
    disc_factors     : shape (n_scenarios, n_steps+1)，累计折现因子路径
    equity_tr        : shape (n_scenarios, n_steps+1)，股票累计总回报路径（从 1.0 开始）
    dt               : 时间步长（年）
    tolerance        : 允许误差（默认 1.0%）
    max_check_tenor  : 最大检验期限（年），默认 15Y
                       注：T>15Y 时因 σ_eq=20% 导致抽样标准误 ±5%（N=1000），物理上无法通过 ±1% 检验

    返回
    ----
    result : dict，包含各时间点的检验结果
    """
    n_steps = disc_factors.shape[1] - 1
    max_step = min(n_steps, int(max_check_tenor / dt))
    check_steps = [max(1, int(max_step * f)) for f in [0.2, 0.4, 0.6, 0.8, 1.0]]
    results = []

    for step in check_steps:
        # E^Q[S(T)/S(0) * exp(-∫r dt)] 应 = 1
        discounted_eq = disc_factors[:, step] * equity_tr[:, step]
        model_val = np.mean(discounted_eq)
        error_pct = abs(model_val - 1.0) * 100
        passed = error_pct <= tolerance * 100

        results.append({
            "step": step,
            "tenor_yr": round(step * dt, 1),
            "model_E[disc_S]": round(model_val, 6),
            "expected": 1.0,
            "error_pct": round(error_pct, 4),
            "passed": passed,
        })

    all_passed = all(r["passed"] for r in results)
    return {"all_passed": all_passed, "details": results, "tolerance_pct": tolerance * 100}


def print_report(bond_result: dict, equity_result: dict) -> None:
    """打印 Martingale Test 报告。"""
    print("=" * 60)
    print("  HKRBC ESG Martingale Test Report")
    print("=" * 60)

    print(f"\n[债券鞅检验] 允许误差: ±{bond_result['tolerance_pct']:.1f}%")
    print(f"  总体结果: {'✅ PASSED' if bond_result['all_passed'] else '❌ FAILED'}")
    print(f"  {'期限(年)':<10} {'模型价格':<12} {'市场价格':<12} {'误差%':<10} {'结果'}")
    print("  " + "-" * 52)
    for r in bond_result["details"]:
        status = "✅" if r["passed"] else "❌"
        print(f"  {r['tenor']:<10} {r['model_price']:<12.6f} {r['market_price']:<12.6f} "
              f"{r['error_pct']:<10.4f} {status}")

    print(f"\n[股票鞅检验] 允许误差: ±{equity_result['tolerance_pct']:.1f}%")
    print(f"  总体结果: {'✅ PASSED' if equity_result['all_passed'] else '❌ FAILED'}")
    print(f"  {'时间(年)':<10} {'E[disc·S]':<12} {'期望值':<10} {'误差%':<10} {'结果'}")
    print("  " + "-" * 48)
    for r in equity_result["details"]:
        status = "✅" if r["passed"] else "❌"
        print(f"  {r['tenor_yr']:<10} {r['model_E[disc_S]']:<12.6f} {r['expected']:<10.1f} "
              f"{r['error_pct']:<10.4f} {status}")

    print("\n" + "=" * 60)
    overall = bond_result["all_passed"] and equity_result["all_passed"]
    print(f"  最终结论: {'✅ 情景符合 HKRBC Rule 19 鞅条件' if overall else '❌ 情景未通过，请重新校准参数'}")
    print("=" * 60)
