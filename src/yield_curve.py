"""
yield_curve.py
==============
根据 IA Schedule 4 零息利率构建即期利率曲线，并提取瞬时远期利率 f(0, t)。
瞬时远期利率是 Hull-White 模型校准至初始收益曲线的关键输入。

HKRBC 监管依据：Cap. 41R Schedule 4；Smith-Wilson 外插法。
"""

import numpy as np
from scipy.interpolate import CubicSpline


def bootstrap_spot_from_par(maturities: np.ndarray, par_rates: np.ndarray) -> np.ndarray:
    """
    从票面利率（par rates）逐步提取零息即期利率（spot rates）。
    采用 bootstrapping 方法，适用于半年付息债券。

    参数
    ----
    maturities : array，年期（年），如 [0.5, 1.0, 1.5, 2.0, ...]
    par_rates  : array，对应票面利率（年化，小数表示），如 [0.03, 0.031, ...]

    返回
    ----
    spot_rates : array，对应零息即期利率（年化，连续复利）
    """
    n = len(maturities)
    spot_rates = np.zeros(n)
    dt = maturities[1] - maturities[0] if n > 1 else maturities[0]

    for i, (T, c) in enumerate(zip(maturities, par_rates)):
        # 票面利率 c 对应每期票息 = c * dt（以 dt 为付息间隔）
        coupon = c * dt
        # 已知前 i 期零息折现因子的现值之和
        pv_coupons = sum(
            coupon * np.exp(-spot_rates[j] * maturities[j])
            for j in range(i)
        )
        # 最后一期：1 元本金 + 票息，令 PV = 1（par bond）
        # (coupon + 1) * exp(-r_i * T_i) = 1 - pv_coupons
        df_i = (1 - pv_coupons) / (1 + coupon)
        spot_rates[i] = -np.log(df_i) / T

    return spot_rates


class YieldCurve:
    """
    基于 IA Schedule 4 离散零息利率，构建连续三次样条插值的即期利率曲线，
    并提供瞬时远期利率 f(0, t) 供 Hull-White 校准使用。

    用法示例
    --------
    maturities = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30])
    spot_rates = np.array([0.030, 0.031, 0.032, 0.034, 0.035, 0.036, 0.037, 0.038, 0.038])
    yc = YieldCurve(maturities, spot_rates)
    f_5 = yc.forward_rate(5.0)   # f(0, 5)
    """

    def __init__(self, maturities: np.ndarray, spot_rates: np.ndarray):
        """
        参数
        ----
        maturities : 到期期限（年），如 [1, 2, 3, 5, 7, 10, 15, 20, 30]
        spot_rates : 对应连续复利零息利率（年化小数），如 IA Sch.4 提供的数值
        """
        # 加入 t=0 端点（零利率）
        if maturities[0] > 0:
            maturities = np.concatenate([[0.0], maturities])
            spot_rates = np.concatenate([[spot_rates[0]], spot_rates])

        self.maturities = maturities
        self.spot_rates = spot_rates
        # 对 r(t)*t（即 log 折现因子的绝对值）做三次样条插值，保证远期利率连续
        self._spline = CubicSpline(maturities, spot_rates * maturities, extrapolate=True)

    def spot_rate(self, t: float) -> float:
        """
        返回 t 年期即期利率 r(0, t)（连续复利）。
        当 t <= 0 时返回短端利率。
        """
        if t <= 1e-8:
            return float(self.spot_rates[1])
        return float(self._spline(t) / t)

    def discount_factor(self, t: float) -> float:
        """返回 t 年期折现因子 P(0, t) = exp(-r(0,t) * t)。"""
        return float(np.exp(-self._spline(t)))

    def forward_rate(self, t: float) -> float:
        """
        返回 t 时刻的瞬时远期利率 f(0, t)。
        定义：f(0, t) = d/dt [r(0,t) * t]
        由三次样条对 r(t)*t 微分得到，连续可微。
        """
        if t <= 1e-8:
            return float(self.spot_rates[1])
        return float(self._spline(t, 1))  # 一阶导数

    def forward_rates_array(self, times: np.ndarray) -> np.ndarray:
        """批量计算瞬时远期利率数组，供 HW1F 校准使用。"""
        result = np.zeros(len(times))
        for i, t in enumerate(times):
            result[i] = self.forward_rate(t)
        return result
