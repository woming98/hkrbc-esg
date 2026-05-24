"""
equity.py
=========
股票总回报模型（Geometric Brownian Motion, GBM），支持 Q-measure 和 P-measure。

── Q-measure（HKRBC TVOG 用）────────────────────────────────────────────
    dS(t) = r(t)·S(t) dt + σ_eq·S(t) dW^Q_eq(t)
    drift = r(t)（无风险利率），确保无套利，martingale test 通过。
    期望年化回报 ≈ r(t)（约 3%–4%，当前 HKD 水平）

── P-measure（ALM/SAA/ORSA 用）──────────────────────────────────────────
    dS(t) = [r(t) + ERP]·S(t) dt + σ_eq·S(t) dW^P_eq(t)
    drift = r(t) + equity_risk_premium（ERP，股票风险溢价）
    ERP 典型值：5%–7%（HK/全球股票历史长期超额回报）
    期望年化回报 ≈ r(t) + ERP ≈ 8%–11%（反映真实世界股票预期回报）
    注意：P-measure 下 martingale test 不成立（这是正常的，不需要通过）

对数离散化（两种 measure 共用，只有 drift 不同）：
    ln S(t+Δt) = ln S(t) + (drift - σ²/2)·Δt + σ·√Δt·Z_eq
"""

import numpy as np


class EquityGBM:
    """
    GBM 股票总回报模型，支持 Q-measure（TVOG）和 P-measure（ALM）。

    参数
    ----
    sigma_eq             : 年化股票波动率，如 HSI 历史波动率约 0.20
    s0                   : 初始指数水平（标准化为 1.0）
    equity_risk_premium  : 股票风险溢价（P-measure 专用，Q-measure 设为 0）
                           典型值：0.055（5.5%，HK 市场历史 ERP 约 5%–7%）
    """

    def __init__(self, sigma_eq: float, s0: float = 1.0, equity_risk_premium: float = 0.0):
        self.sigma_eq = sigma_eq
        self.s0 = s0
        self.equity_risk_premium = equity_risk_premium  # Q-measure: 0.0；P-measure: ~0.055

    def simulate(
        self,
        r_paths: np.ndarray,
        dt: float,
        corr_Z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        模拟股票总回报路径。

        参数
        ----
        r_paths  : shape (n_scenarios, n_steps+1)，来自 HW1F 的短期利率路径
        dt       : 时间步长（年）
        corr_Z   : shape (n_scenarios, n_steps)，Cholesky 变换后的相关随机数

        返回
        ----
        s_paths  : shape (n_scenarios, n_steps+1)，股票指数路径
        tr_paths : shape (n_scenarios, n_steps+1)，累计总回报（从 1.0 开始）
        """
        n_scenarios, n_steps_plus1 = r_paths.shape
        n_steps = n_steps_plus1 - 1
        sigma = self.sigma_eq
        erp = self.equity_risk_premium

        s = np.zeros((n_scenarios, n_steps + 1))
        s[:, 0] = self.s0

        for i in range(n_steps):
            # Q-measure：drift = r(t)；P-measure：drift = r(t) + ERP
            # 使用欧拉法（期初利率）确保 Q-measure 下 disc×S 严格为鞅
            drift = r_paths[:, i] + erp
            log_return = (drift - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * corr_Z[:, i]
            s[:, i + 1] = s[:, i] * np.exp(log_return)

        tr_paths = s / self.s0
        return s, tr_paths
