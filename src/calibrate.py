"""
calibrate.py
============
G2++ (Hull-White 2-Factor) 参数校准模块。

方法：
    利用 G2++ swaption 解析定价公式（1D 数值积分，Brigo & Mercurio 2006 §4.2），
    将模型 ATM normal (Bachelier) vol 与市场 vol 拟合，
    通过 scipy.optimize.differential_evolution 全局优化求解 5 个参数：
        a, b, sigma1, sigma2, rho

数据输入：
    data/swaption_vols_hkd.csv
    字段：expiry_yr, tenor_yr, normal_vol_bps, weight
    可直接替换为 Bloomberg SWPN ATM normal vol 数据。

主要函数：
    calibrate_G2()       -- 主校准入口
    swaption_price_G2()  -- G2++ ATM payer swaption 解析价格
    forward_swap_rate()  -- ATM 远期 swap 利率
    load_vol_surface()   -- 读取 CSV vol surface
"""

import warnings
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq, differential_evolution
from scipy.stats import norm

from src.yield_curve import YieldCurve

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ─────────────────────────────────────────────────────────────────────────────
# G2++ 解析工具
# ─────────────────────────────────────────────────────────────────────────────

def _B(speed: float, t: float, T: float) -> float:
    """B(t,T;a) = (1 - exp(-a*(T-t))) / a，HW/G2++ ZCB 公式中的因子敏感度。"""
    tau = T - t
    if abs(speed) < 1e-9:
        return tau
    return (1.0 - np.exp(-speed * tau)) / speed


def _V_integral(sigma1: float, sigma2: float, rho: float,
                a: float, b: float, t: float, T: float) -> float:
    """
    Var[∫_t^T r(s)ds | F_t] 解析值（Brigo & Mercurio 2006, eq. 4.11）。
    用于计算 G2++ 零息债券价格的 A(t,T) 项。
    """
    tau = T - t

    def h(sig: float, mu: float) -> float:
        if abs(mu) < 1e-9:
            return sig ** 2 * tau ** 3 / 3.0
        return (sig ** 2 / mu ** 2) * (
            tau
            + 2.0 / mu * np.exp(-mu * tau)
            - 1.0 / (2.0 * mu) * np.exp(-2.0 * mu * tau)
            - 3.0 / (2.0 * mu)
        )

    v11 = h(sigma1, a)
    v22 = h(sigma2, b)

    ab = a + b
    if abs(ab) < 1e-9:
        cross = rho * sigma1 * sigma2 * tau ** 3 / 3.0
    else:
        cross = 2.0 * rho * sigma1 * sigma2 / (a * b) * (
            tau
            + (np.exp(-a * tau) - 1.0) / a
            + (np.exp(-b * tau) - 1.0) / b
            - (np.exp(-ab * tau) - 1.0) / ab
        )
    return v11 + v22 + cross


def _zcb_log_params(T0: float, Ti: float,
                    sigma1: float, sigma2: float, rho: float,
                    a: float, b: float,
                    yc: YieldCurve) -> tuple:
    """
    G2++ 中，时刻 T0 到期 Ti 的零息债券价格：
        ln P(T0, Ti) = M + Ba * x(T0) + Bb * y(T0)
    返回 (M, Ba, Bb)，其中 Ba = -B(a,T0,Ti) < 0，Bb = -B(b,T0,Ti) < 0。
    """
    ln_P0Ti = np.log(max(yc.discount_factor(Ti), 1e-15))
    ln_P0T0 = np.log(max(yc.discount_factor(T0), 1e-15))

    V0Ti  = _V_integral(sigma1, sigma2, rho, a, b, 0.0, Ti)
    V0T0  = _V_integral(sigma1, sigma2, rho, a, b, 0.0, T0)
    VT0Ti = _V_integral(sigma1, sigma2, rho, a, b, T0, Ti)

    # A(T0,Ti) = ln[P(0,Ti)/P(0,T0)] - 0.5*(V(0,Ti) - V(0,T0) - V(T0,Ti))
    M  = (ln_P0Ti - ln_P0T0) - 0.5 * (V0Ti - V0T0 - VT0Ti)
    Ba = -_B(a, T0, Ti)  # 负值：x 升高 → 债券价格下降
    Bb = -_B(b, T0, Ti)
    return M, Ba, Bb


# ─────────────────────────────────────────────────────────────────────────────
# Swaption 定价
# ─────────────────────────────────────────────────────────────────────────────

def forward_swap_rate(T0: float, tenor: float, yc: YieldCurve) -> float:
    """
    ATM 远期 swap 利率（年付，act/act 简化）：
        K* = (P(0,T0) - P(0,T0+S)) / Σ P(0,T0+k)
    """
    n = int(round(tenor))
    annuity = sum(yc.discount_factor(T0 + k) for k in range(1, n + 1))
    if annuity < 1e-12:
        return 0.0
    return (yc.discount_factor(T0) - yc.discount_factor(T0 + tenor)) / annuity


def swaption_price_G2(
    T0: float,
    tenor: float,
    K: float,
    sigma1: float,
    sigma2: float,
    rho: float,
    a: float,
    b: float,
    yc: YieldCurve,
) -> float:
    """
    G2++ ATM payer swaption 价格（t=0 时刻），使用 1D 数值积分。

    方法（Brigo & Mercurio 2006, Section 4.2.2）：
        对 x(T0) 边际分布积分，条件给定 x 后对 y 求解 critical y*，
        再解析计算 y < y* 区间的期望值。

    Parameters
    ----------
    T0     : option expiry（年）
    tenor  : swap tenor（年，整数）
    K      : strike（ATM = forward_swap_rate 结果）
    """
    n = int(round(tenor))
    T_pmts = [T0 + k for k in range(1, n + 1)]
    coupons = [K] * (n - 1) + [1.0 + K]  # 年付 coupon，最后期含 principal

    # 预计算各期 ZCB 的解析参数
    zcb_p = [_zcb_log_params(T0, Ti, sigma1, sigma2, rho, a, b, yc) for Ti in T_pmts]

    # x(T0), y(T0) 在 t=0 下的边际方差与协方差
    Vx  = sigma1 ** 2 * (1.0 - np.exp(-2.0 * a * T0)) / (2.0 * a)
    Vy  = sigma2 ** 2 * (1.0 - np.exp(-2.0 * b * T0)) / (2.0 * b)
    Vxy = rho * sigma1 * sigma2 * (1.0 - np.exp(-(a + b) * T0)) / (a + b)

    sx = np.sqrt(max(Vx, 0.0))
    if sx < 1e-12:
        return 0.0

    # y | x 的条件均值和条件方差
    rho_cond   = Vxy / (sx * np.sqrt(max(Vy, 1e-20)))
    vy_cond    = Vy * (1.0 - rho_cond ** 2)
    sy_cond    = np.sqrt(max(vy_cond, 0.0))

    def mu_y_given_x(x0: float) -> float:
        return rho_cond * np.sqrt(Vy) / sx * x0

    def swap_val(x0: float, y0: float) -> float:
        """swap annuity（含 principal）减去 1（即 payer swaption 的基础值）。"""
        return sum(
            coupons[k] * np.exp(zcb_p[k][0] + zcb_p[k][1] * x0 + zcb_p[k][2] * y0)
            for k in range(n)
        ) - 1.0

    def y_star(x0: float) -> float:
        """
        在给定 x0 条件下，使 swap_val = 0 的 y 临界值。
        swap_val 关于 y 单调递减（Bb < 0 → 更大 y → 更低 ZCB 价格）。
        """
        mu = mu_y_given_x(x0)
        span = 6.0 * (sy_cond + 0.01)
        lo, hi = mu - span, mu + span
        flo, fhi = swap_val(x0, lo), swap_val(x0, hi)
        if flo <= 0.0:
            return -np.inf   # 全域 OTM
        if fhi >= 0.0:
            return np.inf    # 全域 ITM
        return brentq(lambda y: swap_val(x0, y), lo, hi, xtol=1e-10, maxiter=100)

    def integrand(u: float) -> float:
        """
        u = x(T0) / sx（标准化）。
        计算 payer swaption payoff 的期望贡献，乘以标准正态密度 φ(u)。
        """
        x0 = u * sx
        mu_y = mu_y_given_x(x0)
        ys   = y_star(x0)

        if ys == -np.inf:
            return 0.0

        result = 0.0
        for k in range(n):
            M, Ba, Bb = zcb_p[k]
            if sy_cond < 1e-12:
                ind = 1.0 if (mu_y < ys or ys == np.inf) else 0.0
                result += coupons[k] * np.exp(M + Ba * x0 + Bb * mu_y) * ind
            else:
                d_star = (ys - mu_y) / sy_cond if ys != np.inf else np.inf
                # E[c_k * P * 1_{y < y*} | x0]，利用 log-normal 矩
                result += (
                    coupons[k]
                    * np.exp(M + Ba * x0 + Bb * mu_y + 0.5 * Bb ** 2 * vy_cond)
                    * norm.cdf(d_star - Bb * sy_cond)
                )

        # 减去 1 * Prob(y < y* | x0) = P(0,T0) 部分
        if sy_cond < 1e-12:
            prob_itm = 1.0 if (mu_y < ys or ys == np.inf) else 0.0
        else:
            d_star = (ys - mu_y) / sy_cond if ys != np.inf else np.inf
            prob_itm = norm.cdf(d_star)

        result -= prob_itm
        return result * norm.pdf(u)

    price, _ = quad(integrand, -6.0, 6.0, limit=120, epsabs=1e-8, epsrel=1e-6)
    return max(price * yc.discount_factor(T0), 0.0)


def price_to_normal_vol(
    T0: float, tenor: float, price: float, yc: YieldCurve
) -> float:
    """
    将 swaption 价格转换为 Bachelier ATM normal vol（单位：decimal/year）。
    ATM Bachelier 公式：price = annuity * sigma_N * sqrt(T0 / (2π))
    """
    n = int(round(tenor))
    annuity = sum(yc.discount_factor(T0 + k) for k in range(1, n + 1))
    denom = annuity * np.sqrt(T0 / (2.0 * np.pi))
    if denom < 1e-14:
        return 0.0
    return price / denom


# ─────────────────────────────────────────────────────────────────────────────
# 主校准函数
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_G2(
    vol_surface: pd.DataFrame,
    yc: YieldCurve,
    bounds: list = None,
    seed: int = 42,
    maxiter: int = 400,
    verbose: bool = True,
) -> dict:
    """
    G2++ 参数校准：将模型 ATM normal vol 拟合至市场 vol surface。

    Parameters
    ----------
    vol_surface : DataFrame，列为 expiry_yr, tenor_yr, normal_vol_bps[, weight]
    yc          : YieldCurve 对象（IA Sch.4 零息曲线）
    bounds      : 5 元素列表 [(a_lo,a_hi), (b_lo,b_hi), ...]，默认合理范围
    seed        : differential_evolution 随机种子
    maxiter     : 最大迭代次数
    verbose     : 是否打印进度

    Returns
    -------
    dict: a, b, sigma1, sigma2, rho, rmse_bps, fitted_vols_bps, converged
    """
    if bounds is None:
        bounds = [
            (0.005, 0.20),   # a：x 因子均值回归速度
            (0.10,  1.00),   # b：y 因子均值回归速度
            (0.002, 0.025),  # sigma1：x 因子波动率
            (0.003, 0.030),  # sigma2：y 因子波动率
            (-0.99, -0.20),  # rho：两因子 Brownian 相关性（负值为主）
        ]

    expiries   = vol_surface["expiry_yr"].values.astype(float)
    tenors     = vol_surface["tenor_yr"].values.astype(float)
    mkt_vols   = vol_surface["normal_vol_bps"].values.astype(float) / 10_000.0
    weights    = (
        vol_surface["weight"].values.astype(float)
        if "weight" in vol_surface.columns
        else np.ones(len(expiries))
    )

    # 预计算 ATM strikes
    K_atm = np.array([forward_swap_rate(T0, S, yc) for T0, S in zip(expiries, tenors)])

    call_count = [0]

    def objective(params: np.ndarray) -> float:
        a, b, s1, s2, rho = params
        loss = 0.0
        call_count[0] += 1
        for T0, S, K, mv, w in zip(expiries, tenors, K_atm, mkt_vols, weights):
            try:
                price = swaption_price_G2(T0, S, K, s1, s2, rho, a, b, yc)
                model_vol = price_to_normal_vol(T0, S, price, yc)
                loss += w * (model_vol - mv) ** 2
            except Exception:
                loss += w * 1.0  # 惩罚数值异常的参数组合
        return loss

    if verbose:
        print("=" * 60)
        print("G2++ 参数校准 — differential_evolution")
        print(f"校准点数：{len(expiries)} 个 ATM swaption")
        print(f"参数边界：a={bounds[0]}, b={bounds[1]}")
        print(f"          σ₁={bounds[2]}, σ₂={bounds[3]}, ρ={bounds[4]}")
        print("=" * 60)

    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=seed,
        maxiter=maxiter,
        tol=1e-9,
        mutation=(0.5, 1.5),
        recombination=0.7,
        popsize=15,
        polish=True,
        disp=verbose,
        workers=1,
    )

    a_opt, b_opt, s1_opt, s2_opt, rho_opt = result.x

    # 计算拟合结果
    fitted_bps = []
    for T0, S, K in zip(expiries, tenors, K_atm):
        try:
            p = swaption_price_G2(T0, S, K, s1_opt, s2_opt, rho_opt, a_opt, b_opt, yc)
            fitted_bps.append(round(price_to_normal_vol(T0, S, p, yc) * 10_000.0, 1))
        except Exception:
            fitted_bps.append(float("nan"))

    mkt_bps = (mkt_vols * 10_000.0).tolist()
    rmse = float(np.sqrt(np.nanmean([(f - m) ** 2 for f, m in zip(fitted_bps, mkt_bps)])))

    if verbose:
        print(f"\n校准结果：")
        print(f"  a      = {a_opt:.6f}   （x 因子均值回归速度）")
        print(f"  b      = {b_opt:.6f}   （y 因子均值回归速度）")
        print(f"  σ₁     = {s1_opt:.6f}   （x 因子波动率）")
        print(f"  σ₂     = {s2_opt:.6f}   （y 因子波动率）")
        print(f"  ρ      = {rho_opt:.6f}   （两因子相关性）")
        print(f"  RMSE   = {rmse:.2f} bps")
        print(f"  收敛   = {result.success}  （共 {call_count[0]} 次函数评估）")
        print()
        print(f"{'Expiry':>8} {'Tenor':>7} {'Market':>10} {'Model':>10} {'Error':>9}")
        print("-" * 50)
        for T0, S, mv, fv in zip(expiries, tenors, mkt_bps, fitted_bps):
            err = fv - mv if not np.isnan(fv) else float("nan")
            print(f"{T0:>6.0f}Y  {S:>5.0f}Y  {mv:>9.1f}  {fv:>9.1f}  {err:>+8.1f}")

    return {
        "a":               round(a_opt,   6),
        "b":               round(b_opt,   6),
        "sigma1":          round(s1_opt,  6),
        "sigma2":          round(s2_opt,  6),
        "rho":             round(rho_opt, 6),
        "rmse_bps":        round(rmse,    3),
        "fitted_vols_bps": fitted_bps,
        "market_vols_bps": mkt_bps,
        "expiries":        expiries.tolist(),
        "tenors":          tenors.tolist(),
        "converged":       result.success,
        "n_func_evals":    call_count[0],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 辅助工具
# ─────────────────────────────────────────────────────────────────────────────

def load_vol_surface(csv_path: str) -> pd.DataFrame:
    """
    从 CSV 读取 swaption vol surface。
    跳过以 '#' 开头的注释行；必要列：expiry_yr, tenor_yr, normal_vol_bps。
    """
    df = pd.read_csv(csv_path, comment="#")
    required = {"expiry_yr", "tenor_yr", "normal_vol_bps"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少以下列：{missing}")
    return df
