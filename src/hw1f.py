"""
hw1f.py
=======
Hull-White 单因子（HW1F）利率模型，支持 Q-measure（TVOG）和 P-measure（ALM/ORSA）。

── Q-measure（Risk-Neutral，HKRBC TVOG 用）──────────────────────────────
    dr(t) = [θ(t) - a·r(t)] dt + σ·dW^Q(t)
    θ(t) 由 IA Sch.4 yield curve 唯一确定：
      θ(t) = ∂f(0,t)/∂t + a·f(0,t) + σ²/(2a)·(1-e^{-2at})
    特征：精确拟合初始 yield curve，通过 martingale test，用于 TVOG 定价。

── P-measure（Real-World，ALM/SAA/ORSA 用）──────────────────────────────
    dr(t) = [θ_P(t) - a·r(t)] dt + σ·dW^P(t)
    θ_P(t) = θ(t) + a·λ(t)   其中 λ(t) = term_premium（利率风险溢价）
    实际效果：长端均值 = forward rate + term_premium（约 +0.5–1.5%）
    特征：长期利率比 Q-measure 更高（反映真实世界投资者要求的利率风险补偿）。
    不通过 martingale test（P-measure 下期望值 ≠ 市场价格，这是正常的）。

精确离散化（两种 measure 共用）：
    r(t+Δt) | r(t) ~ Normal(μ(t), v²(Δt))
    μ(t)   = r(t)·e^{-aΔt} + α_eff(t+Δt) - α_eff(t)·e^{-aΔt}
    v²(Δt) = σ²/(2a)·(1 - e^{-2aΔt})

HKRBC 要求：Q-measure ≥1,000 条，market-consistent，无套利（martingale test）。
"""

import numpy as np
from src.yield_curve import YieldCurve


class HullWhite1F:
    """
    Hull-White 1-Factor 利率模型，支持 Q-measure 和 P-measure。

    参数
    ----
    a            : float，均值回归速度，通常 0.01–0.20
    sigma        : float，利率波动率，通常 0.005–0.025
    yc           : YieldCurve，初始 IA Schedule 4 yield curve 对象
    term_premium : float，利率期限溢价（P-measure 专用，Q-measure 设为 0）
                   典型值：HKD 约 0.005–0.015（0.5%–1.5%）
    """

    def __init__(self, a: float, sigma: float, yc: YieldCurve, term_premium: float = 0.0):
        self.a = a
        self.sigma = sigma
        self.yc = yc
        self.term_premium = term_premium  # Q-measure: 0.0；P-measure: 约 0.008

    def alpha(self, t: float) -> float:
        """
        辅助函数 α_eff(t)：精确离散化的长期均值调整项。

        Q-measure：α(t) = f(0,t) + σ²/(2a²)·(1-e^{-at})²
        P-measure：α_P(t) = f(0,t) + term_premium + σ²/(2a²)·(1-e^{-at})²
                   即在 Q-measure 基础上加一个常数 term_premium，
                   使长端利率均值高于市场 forward rate，反映真实世界期望。
        """
        a, sigma = self.a, self.sigma
        f0t = self.yc.forward_rate(t)
        return f0t + self.term_premium + (sigma ** 2 / (2 * a ** 2)) * (1 - np.exp(-a * t)) ** 2

    def simulate(
        self,
        n_scenarios: int,
        n_steps: int,
        dt: float,
        seed: int = 42,
        Z_external: np.ndarray = None,
        antithetic: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        模拟利率路径。

        参数
        ----
        n_scenarios : 情景数量，HKRBC 要求 ≥ 1,000
        n_steps     : 时间步数（月数），如 360 = 30年×12月
        dt          : 时间步长（年），如 1/12 为月度步长
        seed        : 随机种子，确保结果可重现
        Z_external  : shape (n_scenarios, n_steps)，外部随机数（用于与其他因子联动相关性）
                      若为 None 则内部生成
        antithetic  : 是否使用对偶变量法（antithetic variates）降低方差，提升 martingale test 通过率

        返回
        ----
        r_paths      : shape (n_scenarios, n_steps+1)，短期利率路径
        disc_factors : shape (n_scenarios, n_steps+1)，累计折现因子 exp(-∫r dt)
        """
        a, sigma = self.a, self.sigma

        r = np.zeros((n_scenarios, n_steps + 1))
        disc = np.ones((n_scenarios, n_steps + 1))

        # 初始短期利率 r(0) = f(0, 0+)
        r0 = self.yc.forward_rate(1e-6)
        r[:, 0] = r0

        # 精确离散化条件方差（常数）
        var_r = (sigma ** 2 / (2 * a)) * (1 - np.exp(-2 * a * dt))
        std_r = np.sqrt(var_r)

        # 准备随机数
        if Z_external is not None:
            # 使用外部传入的相关随机数（已 Cholesky 变换）
            Z_all = Z_external  # shape (n_scenarios, n_steps)
        else:
            np.random.seed(seed)
            if antithetic:
                # 对偶变量法：前半部分用 Z，后半部分用 -Z，大幅降低 martingale 方差
                half = n_scenarios // 2
                Z_half = np.random.standard_normal((half, n_steps))
                Z_all = np.concatenate([Z_half, -Z_half], axis=0)
            else:
                Z_all = np.random.standard_normal((n_scenarios, n_steps))

        # 预计算各时间步的 alpha 值（向量化提速）
        times = np.array([i * dt for i in range(n_steps + 1)])
        alphas = np.array([self.alpha(t) for t in times])

        e_dt = np.exp(-a * dt)

        for i in range(n_steps):
            # 精确离散化条件均值
            mean_r = r[:, i] * e_dt + alphas[i + 1] - alphas[i] * e_dt
            r[:, i + 1] = mean_r + std_r * Z_all[:, i]

            # 欧拉近似折现因子：使用期初利率 r[i]
            # 与 equity.py 的漂移项保持一致（均用 r[i]），确保 disc×S 的乘积严格为鞅
            disc[:, i + 1] = disc[:, i] * np.exp(-r[:, i] * dt)

        return r, disc

    def zero_coupon_bond_price(
        self, r_t: np.ndarray, t: float, T: float
    ) -> np.ndarray:
        """
        HW1F 解析债券价格公式 P(t, T)，用于 martingale test 验证。

        P(t, T) = A(t, T) · exp(-B(t, T) · r(t))

        参数
        ----
        r_t : 当前短期利率（可为数组，对应各情景）
        t   : 当前时刻
        T   : 债券到期时刻

        返回
        ----
        P(t, T) 的解析值
        """
        a, sigma = self.a, self.sigma
        tau = T - t
        B = (1 - np.exp(-a * tau)) / a
        ln_A = (
            np.log(self.yc.discount_factor(T) / self.yc.discount_factor(t))
            + B * self.yc.forward_rate(t)
            - (sigma ** 2 / (4 * a)) * B ** 2 * (1 - np.exp(-2 * a * t))
        )
        return np.exp(ln_A - B * r_t)
