"""
equity.py
=========
Q-measure 下的股票总回报模型（Geometric Brownian Motion, GBM）。

Q-measure 下股票 SDE：
    dS(t) = r(t)·S(t) dt + σ_eq·S(t) dW^Q_eq(t)

其中 drift = r(t)（风险中性条件，无套利），σ_eq 为股票年化波动率。

对数离散化（精确）：
    ln S(t+Δt) = ln S(t) + (r(t) - σ²/2)·Δt + σ·√Δt·Z_eq

相关性结构：利率冲击 Z_r 与股票冲击 Z_eq 通过 Cholesky 分解处理，
见 cholesky_corr.py。

HKRBC 注意事项：
- 使用历史隐含波动率（通常 HSI 30天实现波动率或期权隐含波动率）
- 香港市场参考 HSI 指数
"""

import numpy as np


class EquityGBM:
    """
    Q-measure GBM 股票模型，驱动利率来自 HW1F 路径。

    参数
    ----
    sigma_eq : 年化股票波动率（小数），如 HSI 历史波动率约 0.20
    s0       : 初始股票指数水平（通常标准化为 1.0）
    """

    def __init__(self, sigma_eq: float, s0: float = 1.0):
        self.sigma_eq = sigma_eq
        self.s0 = s0

    def simulate(
        self,
        r_paths: np.ndarray,
        dt: float,
        corr_Z: np.ndarray,
    ) -> np.ndarray:
        """
        模拟股票总回报路径。

        参数
        ----
        r_paths  : shape (n_scenarios, n_steps+1)，来自 HW1F 的短期利率路径
        dt       : 时间步长（年）
        corr_Z   : shape (n_scenarios, n_steps)，与利率相关的标准正态随机数
                   （已经过 Cholesky 变换）

        返回
        ----
        s_paths     : shape (n_scenarios, n_steps+1)，股票指数路径（S(t)/S(0)）
        tr_paths    : shape (n_scenarios, n_steps+1)，累计总回报（从 1.0 开始）
        """
        n_scenarios, n_steps_plus1 = r_paths.shape
        n_steps = n_steps_plus1 - 1
        sigma = self.sigma_eq

        s = np.zeros((n_scenarios, n_steps + 1))
        s[:, 0] = self.s0

        for i in range(n_steps):
            # 使用当期利率（欧拉法）作为 Q-measure 漂移项
            # 避免用 r[i+1]（未来值）产生 look-ahead 偏差，改善 martingale 精度
            log_return = (r_paths[:, i] - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * corr_Z[:, i]
            s[:, i + 1] = s[:, i] * np.exp(log_return)

        # 总回报（包含再投资股息，此处简化为纯价格回报，可扩展加股息率）
        tr_paths = s / self.s0
        return s, tr_paths
