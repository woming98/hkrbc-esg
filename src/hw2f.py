"""
hw2f.py
=======
Hull-White 2-Factor (G2++) 利率模型。

模型结构：
    r(t) = x(t) + y(t) + φ(t)

    dx(t) = −a·x(t) dt + σ₁ dW₁(t)
    dy(t) = −b·y(t) dt + σ₂ dW₂(t)
    dW₁·dW₂ = ρ_xy dt

φ(t) 吸收初始 yield curve，使模型精确拟合 IA Sch.4：
    φ(t) = f(0,t) + term_premium
           + σ₁²/(2a²)·(1−e^{−at})²
           + σ₂²/(2b²)·(1−e^{−bt})²
           + ρ_xy·σ₁σ₂/(ab)·(1−e^{−at})·(1−e^{−bt})

精确离散化（避免 Euler 误差）：
    x(t+Δt) = x(t)·e^{−aΔt} + σ₁·v₁(Δt)·Z₁
    y(t+Δt) = y(t)·e^{−bΔt} + σ₂·v₂(Δt)·Z₂
    v₁(Δt)  = √((1−e^{−2aΔt})/(2a))
    v₂(Δt)  = √((1−e^{−2bΔt})/(2b))
    Z₁, Z₂ 相关性通过内部 Cholesky 处理（ρ_xy）

优势 vs HW1F：
    - HW1F：yield curve 只能平行移动
    - HW2F：x 驱动"水平"，y 驱动"斜率"，可模拟
            bear steepening、bull flattening 等真实利率形态
    - Bond/Swaption 仍有解析公式（G2++ 闭合解）

参数典型值（HKD 市场）：
    a=0.05, b=0.50, σ₁=0.010, σ₂=0.015, ρ_xy=−0.80
"""

import numpy as np
from src.yield_curve import YieldCurve


class HullWhite2F:
    """
    G2++ 双因子利率模型。

    参数
    ----
    a            : x 因子均值回归速度（慢因子，驱动水平，典型 0.01–0.10）
    b            : y 因子均值回归速度（快因子，驱动斜率，典型 0.20–0.80）
    sigma1       : x 因子波动率
    sigma2       : y 因子波动率
    rho_xy       : x,y 因子相关性（典型 −0.70 至 −0.90）
    yc           : YieldCurve 对象
    term_premium : P-measure 期限溢价（Q-mode 设 0）
    """

    def __init__(
        self,
        a: float,
        b: float,
        sigma1: float,
        sigma2: float,
        rho_xy: float,
        yc: YieldCurve,
        term_premium: float = 0.0,
    ):
        self.a = a
        self.b = b
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.rho_xy = rho_xy
        self.yc = yc
        self.term_premium = term_premium

    def phi(self, t: float) -> float:
        """
        G2++ 的 yield curve 拟合项（对应 HW1F 的 alpha(t)）。
        φ(t) 确保 E^Q[r(t)] = f(0,t) + term_premium。
        """
        a, b = self.a, self.b
        s1, s2, rho = self.sigma1, self.sigma2, self.rho_xy
        f0t = self.yc.forward_rate(t)
        term1 = (s1 ** 2 / (2 * a ** 2)) * (1 - np.exp(-a * t)) ** 2
        term2 = (s2 ** 2 / (2 * b ** 2)) * (1 - np.exp(-b * t)) ** 2
        term3 = rho * s1 * s2 / (a * b) * (1 - np.exp(-a * t)) * (1 - np.exp(-b * t))
        return f0t + self.term_premium + term1 + term2 + term3

    def simulate(
        self,
        n_scenarios: int,
        n_steps: int,
        dt: float,
        seed: int = 42,
        Z_external: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        精确离散化模拟 G2++ 路径。

        参数
        ----
        n_scenarios : 情景数
        n_steps     : 时间步数
        dt          : 时间步长（年）
        seed        : 随机种子（Z_external 为 None 时生效）
        Z_external  : shape (n_scenarios, n_steps)，外部主 IR 冲击（来自 Cholesky）

        返回
        ----
        r_paths     : shape (n_scenarios, n_steps+1)，短期利率
        disc_factors: shape (n_scenarios, n_steps+1)，累计折现因子
        x_paths     : shape (n_scenarios, n_steps+1)，x 因子路径
        y_paths     : shape (n_scenarios, n_steps+1)，y 因子路径
        """
        a, b = self.a, self.b
        s1, s2, rho = self.sigma1, self.sigma2, self.rho_xy
        rng = np.random.default_rng(seed)

        # 精确离散化的条件标准差
        v1 = np.sqrt((1 - np.exp(-2 * a * dt)) / (2 * a))
        v2 = np.sqrt((1 - np.exp(-2 * b * dt)) / (2 * b))

        # x, y 因子内部 Cholesky（保持 ρ_xy 相关性）
        # Z_x = Z_external（来自外部 Cholesky，代表整体 IR 方向）
        # Z_y = ρ_xy·Z_x + √(1−ρ_xy²)·Z_independent
        if Z_external is None:
            rng = np.random.default_rng(seed)
            Z_x = rng.standard_normal((n_scenarios, n_steps))
        else:
            Z_x = Z_external

        # 对 Z_ind 也用对偶变量法，保持与 Z_x 相同的对称结构
        half = n_scenarios // 2
        Z_ind_half = np.random.default_rng(seed + 99).standard_normal((half, n_steps))
        Z_ind = np.concatenate([Z_ind_half, -Z_ind_half], axis=0)
        Z_y = rho * Z_x + np.sqrt(1 - rho ** 2) * Z_ind

        x = np.zeros((n_scenarios, n_steps + 1))
        y = np.zeros((n_scenarios, n_steps + 1))
        r = np.zeros((n_scenarios, n_steps + 1))
        disc = np.zeros((n_scenarios, n_steps + 1))
        disc[:, 0] = 1.0

        # r(0) = x(0) + y(0) + φ(0) = 0 + 0 + f(0,0)
        r[:, 0] = self.phi(0)

        for i in range(n_steps):
            t_next = (i + 1) * dt
            # 精确离散化
            x[:, i + 1] = x[:, i] * np.exp(-a * dt) + s1 * v1 * Z_x[:, i]
            y[:, i + 1] = y[:, i] * np.exp(-b * dt) + s2 * v2 * Z_y[:, i]
            r[:, i + 1] = x[:, i + 1] + y[:, i + 1] + self.phi(t_next)
            # Euler 折现因子（与 HW1F 保持一致）
            disc[:, i + 1] = disc[:, i] * np.exp(-r[:, i] * dt)

        return r, disc, x, y
