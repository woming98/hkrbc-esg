"""
property_return.py
==================
房地产总回报模型：以股票回报为代理（Proxy to Equity），加时间滞后。

原理：
    直接房地产市场缺乏流动性期权数据，无法直接建立随机 vol 模型。
    实践中用股票指数代理房地产周期，加入：
      1. 时间滞后（lag）：房地产对股市反应慢 1–2 年
      2. 缩放系数（beta）：房地产波动率通常低于股票（约 0.5–0.8×）
      3. 独立波动（idio_vol）：特有的市场摩擦（流动性不足、估值滞后）

模型：
    对数回报：
    ln_R_prop(t) = beta × ln_R_eq(t − lag) + sigma_idio × √Δt × ε(t)

    ε(t) ~ N(0,1)，与 ln_R_eq 独立（代表房地产特有噪声）

    累计房地产回报（以 1.0 为基准）：
    prop_index(t+1) = prop_index(t) × exp(ln_R_prop(t))

HK 参数参考（香港商业房地产 REIT 历史）：
    lag       ≈ 4 步（月步长时 = 4个月；年步长时 = 1年）
    beta      ≈ 0.60（HSI 动1%，商业房地产约动 0.6%）
    sigma_idio ≈ 0.08（特有年化波动率约 8%，流动性溢价）

注意：
    - Property 不参与 martingale test（无可校准市场期权）
    - 本模型适用于 ALM 压力测试；HKRBC TVOG 中 property 占比通常不大
    - 若 Par Fund 有大量物业持仓，建议用专业 IPD/MSCI 物业指数校准
"""

import numpy as np


class PropertyReturn:
    """
    房地产代理模型：以时间滞后的股票回报代理房地产回报。

    参数
    ----
    beta       : 股票-房地产敏感度（典型 0.5–0.8）
    lag_steps  : 股票回报的时间滞后步数（如月步长时 lag=4 表示滞后4个月）
    sigma_idio : 独立波动（年化），房地产特有风险
    prop0      : 初始房地产指数（标准化为 1.0）
    """

    def __init__(
        self,
        beta: float = 0.60,
        lag_steps: int = 4,
        sigma_idio: float = 0.08,
        prop0: float = 1.0,
    ):
        self.beta = beta
        self.lag_steps = lag_steps
        self.sigma_idio = sigma_idio
        self.prop0 = prop0

    def simulate(
        self,
        eq_log_returns: np.ndarray,
        dt: float,
        seed: int = 123,
    ) -> np.ndarray:
        """
        基于股票对数回报序列，模拟房地产总回报路径。

        参数
        ----
        eq_log_returns : shape (n_scenarios, n_steps)，股票逐步对数回报
                         eq_log_return[s, i] = ln(S[s,i+1]/S[s,i])
        dt             : 时间步长（年）
        seed           : 随机种子

        返回
        ----
        prop_paths : shape (n_scenarios, n_steps+1)，房地产累计回报指数（从 1.0 开始）
        """
        n_scenarios, n_steps = eq_log_returns.shape
        rng = np.random.default_rng(seed)

        # 独立噪声
        Z_idio = rng.standard_normal((n_scenarios, n_steps))

        prop = np.zeros((n_scenarios, n_steps + 1))
        prop[:, 0] = self.prop0

        for i in range(n_steps):
            # 滞后股票回报：若 i < lag，用 t=0 的回报（近似为 0）
            lag_i = i - self.lag_steps
            if lag_i >= 0:
                eq_lag = eq_log_returns[:, lag_i]
            else:
                eq_lag = np.zeros(n_scenarios)

            ln_ret = (self.beta * eq_lag
                      + self.sigma_idio * np.sqrt(dt) * Z_idio[:, i])
            prop[:, i + 1] = prop[:, i] * np.exp(ln_ret)

        return prop
