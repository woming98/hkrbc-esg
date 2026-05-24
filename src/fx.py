"""
fx.py
=====
Garman-Kohlhagen FX 模型（GBM + 利率平价）。

Q-measure 下的 FX SDE（连续复利）：
    dS_fx(t) = (r_d(t) − r_f) · S_fx(t) dt + σ_fx · S_fx(t) dW_fx(t)

    S_fx(t) ：外币 / 本币汇率（如 1 HKD = S_fx USD，初始值 S0=1.0）
    r_d(t)  ：本币无风险利率（来自 HW1F/HW2F 路径，随机的）
    r_f     ：外币无风险利率（固定假设，如 USD 利率约 4.5%）
    σ_fx    ：FX 年化波动率

利率平价（Covered Interest Rate Parity）：
    Q-measure 下 drift = r_d(t) − r_f，确保外币折现资产为鞅。
    即：E^Q[disc_d(T) · S_fx(T)] = S_fx(0)  ←  Martingale 性质。

HK 市场参考值：
    HKD/USD：联系汇率制（σ_fx ≈ 0.1%，几乎零波动）
    HKD/RMB：σ_fx ≈ 2%–4%（CNH 波动率）
    HKD/EUR：σ_fx ≈ 6%–8%

对数离散化（精确）：
    ln S_fx(t+Δt) = ln S_fx(t) + (r_d(t) − r_f − 0.5σ²)·Δt + σ·√Δt·Z_fx
"""

import numpy as np


class GarmanKohlhagen:
    """
    Garman-Kohlhagen FX 模型，Q-measure。

    参数
    ----
    sigma_fx : FX 年化波动率（如 0.001=HKD/USD，0.03=HKD/RMB）
    r_f      : 外币无风险利率（固定，年化，如 USD SOFR ≈ 0.045）
    s0       : 初始汇率（标准化为 1.0）
    """

    def __init__(self, sigma_fx: float, r_f: float = 0.0, s0: float = 1.0):
        self.sigma_fx = sigma_fx
        self.r_f = r_f
        self.s0 = s0

    def simulate(
        self,
        r_d_paths: np.ndarray,
        dt: float,
        corr_Z: np.ndarray,
    ) -> np.ndarray:
        """
        模拟 FX 汇率路径。

        参数
        ----
        r_d_paths : shape (n_scenarios, n_steps+1)，本币利率路径（来自 HW1F/HW2F）
        dt        : 时间步长（年）
        corr_Z    : shape (n_scenarios, n_steps)，已 Cholesky 变换的相关随机数

        返回
        ----
        fx_paths : shape (n_scenarios, n_steps+1)，汇率路径（以 s0=1.0 为基准）
        """
        n_scenarios, n_steps_plus1 = r_d_paths.shape
        n_steps = n_steps_plus1 - 1
        sigma = self.sigma_fx
        r_f = self.r_f

        fx = np.zeros((n_scenarios, n_steps + 1))
        fx[:, 0] = self.s0

        for i in range(n_steps):
            # drift = r_d(t) − r_f（利率平价，Euler 法）
            drift = r_d_paths[:, i] - r_f
            log_ret = (drift - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * corr_Z[:, i]
            fx[:, i + 1] = fx[:, i] * np.exp(log_ret)

        return fx
