"""
cir.py
======
Cox-Ingersoll-Ross (CIR) 过程，用于信用利差建模。

CIR SDE：
    dcs(t) = κ·(μ − cs(t)) dt + σ·√cs(t) dW(t)

关键性质：
    - cs(t) ≥ 0 严格正值（满足 Feller 条件 2κμ > σ² 时不触零）
    - 均值回归：长期均值为 μ
    - 方差随 cs 水平成比例，cs 越大波动越大（更符合信用利差实际行为）
    - 相比 OU 过程：OU 可产生负利差，CIR 确保非负

离散化（Euler-Maruyama，带反射边界）：
    cs(t+Δt) = cs(t) + κ·(μ−cs(t))·Δt + σ·√max(cs(t),0)·√Δt·Z
    cs(t+Δt) = max(cs(t+Δt), 0)   ← 反射边界确保非负

注意：Milstein 离散化精度更高，但 Euler + clip 对 HKRBC 场景已足够。

HKRBC 信用利差参考值（HKD IG，2024）：
    cs₀ ≈ 85bps，μ ≈ 80bps，κ ≈ 0.15，σ ≈ 0.025
    Feller 条件：2×0.15×0.008 = 0.0024 > σ² = 0.000625 ✅
"""

import numpy as np


class CIRProcess:
    """
    Cox-Ingersoll-Ross 过程，用于信用利差随机建模。

    参数
    ----
    kappa : 均值回归速度（典型值 0.10–0.30）
    mu    : 长期均值（信用利差年化，如 0.008 = 80bps）
    sigma : 波动率系数（典型值 0.015–0.035）
    cs0   : 初始信用利差（如 0.0085 = 85bps）
    """

    def __init__(self, kappa: float, mu: float, sigma: float, cs0: float):
        self.kappa = kappa
        self.mu = mu
        self.sigma = sigma
        self.cs0 = cs0

        # Feller 条件校验：2κμ > σ² 时过程严格正
        feller = 2 * kappa * mu
        sigma_sq = sigma ** 2
        self._feller_satisfied = feller > sigma_sq
        if not self._feller_satisfied:
            print(f"[CIR] ⚠️  Feller 条件不满足：2κμ={feller:.6f} ≤ σ²={sigma_sq:.6f}")
            print(f"         利差路径可能触及 0，已加反射边界确保非负。")

    def simulate(
        self,
        n_scenarios: int,
        n_steps: int,
        dt: float,
        corr_Z: np.ndarray,
    ) -> np.ndarray:
        """
        模拟 CIR 信用利差路径。

        参数
        ----
        n_scenarios : 情景数
        n_steps     : 时间步数
        dt          : 时间步长（年）
        corr_Z      : shape (n_scenarios, n_steps)，已经过 Cholesky 变换的相关随机数

        返回
        ----
        cs_paths : shape (n_scenarios, n_steps+1)，信用利差路径（年化小数）
        """
        cs = np.zeros((n_scenarios, n_steps + 1))
        cs[:, 0] = self.cs0

        kappa, mu, sigma = self.kappa, self.mu, self.sigma
        sqrt_dt = np.sqrt(dt)

        for i in range(n_steps):
            cs_cur = np.maximum(cs[:, i], 0.0)  # 确保 √cs 有定义
            drift = kappa * (mu - cs_cur) * dt
            diffusion = sigma * np.sqrt(cs_cur) * sqrt_dt * corr_Z[:, i]
            cs[:, i + 1] = np.maximum(cs_cur + drift + diffusion, 0.0)

        return cs
