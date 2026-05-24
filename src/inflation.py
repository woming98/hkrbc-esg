"""
inflation.py
============
通货膨胀随机过程（OU / 简化 Jarrow-Yildirim）。

模型 A（默认，OU 过程）：
    dπ(t) = κ_π·(μ_π − π(t)) dt + σ_π·dW_π(t)

    π(t)  ：瞬时通胀率（年化）
    κ_π   ：均值回归速度（通胀有很强的均值回归特性）
    μ_π   ：长期通胀目标（HK CPI 长期约 2.5%–3.5%）
    σ_π   ：通胀波动率（HK 历史约 1%–2%）

CPI 指数路径：
    CPI(t) = CPI(0) × exp(∫₀ᵗ π(s) ds)
    离散化：CPI(t+Δt) = CPI(t) × exp(π(t)·Δt)

相关性：
    通胀与利率正相关（ρ_ir_infl ≈ 0.30–0.50）
    通胀与股票相关性不稳定（短期负相关，长期近零）
    本模型将通胀视为独立因子（简化处理），可通过 corr_Z 引入相关性

HK 参数参考（基于 C&SD 月度 CPI，2005–2024）：
    μ_π    ≈ 0.030（3.0%，HK 长期通胀中枢）
    κ_π    ≈ 0.20（均值回归速度，相当于约 5 年半衰期）
    σ_π    ≈ 0.015（波动率 1.5%）
    π(0)   ≈ 0.020（当前通胀约 2%，2024年）

应用场景：
    - 费用风险建模（Cap. 41R Rule 5 经营费用假设）
    - 长期护理险、年金通胀挂钩受益
    - ORSA 通胀情景压力测试
"""

import numpy as np


class InflationOU:
    """
    通货膨胀 OU 均值回归过程。

    参数
    ----
    kappa_pi : 均值回归速度（典型 0.10–0.30）
    mu_pi    : 长期通胀均值（年化，如 0.030 = 3%）
    sigma_pi : 通胀波动率（年化，如 0.015 = 1.5%）
    pi0      : 初始通胀率（年化，如 0.020 = 2%）
    """

    def __init__(
        self,
        kappa_pi: float = 0.20,
        mu_pi: float = 0.030,
        sigma_pi: float = 0.015,
        pi0: float = 0.020,
    ):
        self.kappa_pi = kappa_pi
        self.mu_pi = mu_pi
        self.sigma_pi = sigma_pi
        self.pi0 = pi0

    def simulate(
        self,
        n_scenarios: int,
        n_steps: int,
        dt: float,
        corr_Z: np.ndarray = None,
        seed: int = 777,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        模拟通胀率路径和 CPI 指数路径。

        参数
        ----
        n_scenarios : 情景数
        n_steps     : 时间步数
        dt          : 时间步长（年）
        corr_Z      : shape (n_scenarios, n_steps)，外部相关随机数（可选）
                      若为 None，则生成独立随机数
        seed        : 随机种子（corr_Z=None 时生效）

        返回
        ----
        pi_paths  : shape (n_scenarios, n_steps+1)，通胀率路径（年化）
        cpi_paths : shape (n_scenarios, n_steps+1)，CPI 指数路径（以 1.0 为基准）
        """
        if corr_Z is None:
            rng = np.random.default_rng(seed)
            corr_Z = rng.standard_normal((n_scenarios, n_steps))

        kappa, mu, sigma = self.kappa_pi, self.mu_pi, self.sigma_pi
        sqrt_dt = np.sqrt(dt)

        pi = np.zeros((n_scenarios, n_steps + 1))
        cpi = np.zeros((n_scenarios, n_steps + 1))
        pi[:, 0] = self.pi0
        cpi[:, 0] = 1.0

        for i in range(n_steps):
            drift = kappa * (mu - pi[:, i]) * dt
            diffusion = sigma * sqrt_dt * corr_Z[:, i]
            pi[:, i + 1] = pi[:, i] + drift + diffusion
            # CPI 指数累积：exp(π·Δt)
            cpi[:, i + 1] = cpi[:, i] * np.exp(pi[:, i] * dt)

        return pi, cpi
