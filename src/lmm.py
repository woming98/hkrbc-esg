"""
lmm.py
======
LIBOR Market Model (LMM / BGM) 简化实现。

模型核心思想：
    直接对每个 tenor 的 forward rate L_i(t) 建模，而非短期利率 r(t)。
    i = 0, 1, ..., N-1 代表 tenor：1Y, 2Y, ..., N年

Q-measure（Spot Measure）下的 SDE：
    dL_i(t) = L_i(t) · μ_i^spot(t) · dt + L_i(t) · σ_i · dW_i(t)

    μ_i^spot(t) = Σ_{k=p(t)}^{i} [δ·σ_k·σ_i·ρ_ki·L_k(t)] / [1 + δ·L_k(t)]
    p(t) = 当前时间 t 对应的最近 tenor 索引

    ρ_ki = exp(−λ·|k−i|)    （指数衰减相关结构，Schoenmakers-Coffey 简化）

短期利率近似：
    r(t) ≈ L_{p(t)}(t)    （用当期最短 live forward rate 近似）

折现因子：
    disc(T) = Π_{i=0}^{p(T)-1} 1/(1 + δ·L_i(t_i))    （离散复利）

初始 forward rate 校准：
    L_i(0) 由 IA Sch.4 spot rates 推导：
    (1 + δ·L_i(0)) = P(0, iδ) / P(0, (i+1)δ)
    其中 P(0,t) = exp(−s(t)·t)，s(t) 为 spot rate

参数（简化版，全 tenor 使用同一波动率函数）：
    σ_i = σ_base · exp(−decay·i·δ)   （短端波动率高，长端低）
    λ    ：相关衰减参数（典型 0.05–0.20）

注意：
    本实现为 Euler-Maruyama 离散化，步长 Δt = δ（tenor 步长）。
    对于 HKRBC TVOG 应用，HW1F/HW2F 通常已足够，LMM 适用于
    需要精确建模整个 vol surface（多 strike × 多 tenor）的复杂保证产品。
"""

import numpy as np
from src.yield_curve import YieldCurve


class LMM:
    """
    简化版 LIBOR Market Model（BGM），Spot Measure 下 Euler 离散化。

    参数
    ----
    tenors       : list[float]，tenor 网格（年），如 [1,2,...,30]
    sigma_vols   : list[float]，各 tenor 的 forward rate vol（年化），
                   或单一 float（全 tenor 使用相同 vol）
    rho_decay    : float，相关衰减参数 λ，ρ_{ij}=exp(−λ|i−j|)，典型 0.10
    yc           : YieldCurve，初始收益率曲线（用于校准 L_i(0)）
    """

    def __init__(
        self,
        tenors: list,
        sigma_vols,
        rho_decay: float,
        yc: YieldCurve,
    ):
        self.tenors = np.array(tenors, dtype=float)
        self.n_tenors = len(tenors)
        self.delta = tenors[0]  # 假设等间距 tenor 步长（年）

        # 各 tenor 的 vol
        if np.isscalar(sigma_vols):
            self.sigma_vols = np.full(self.n_tenors, float(sigma_vols))
        else:
            self.sigma_vols = np.array(sigma_vols, dtype=float)

        self.rho_decay = rho_decay
        self.yc = yc

        # 构建 tenor 相关矩阵（Schoenmakers-Coffey 指数衰减）
        self.rho_matrix = self._build_correlation()

        # Cholesky 分解相关矩阵（用于生成相关随机数）
        self.chol = np.linalg.cholesky(self.rho_matrix)

        # 初始 forward rates（由 IA Sch.4 推导）
        self.L0 = self._calibrate_initial_forwards()

    def _build_correlation(self) -> np.ndarray:
        """构建 N×N 相关矩阵：ρ_{ij} = exp(−λ|i−j|)。"""
        idx = np.arange(self.n_tenors)
        diff = np.abs(idx[:, None] - idx[None, :])
        return np.exp(-self.rho_decay * diff)

    def _calibrate_initial_forwards(self) -> np.ndarray:
        """
        从 IA Sch.4 spot rates 推导初始 LIBOR forward rates。
        L_i(0) = (P(0, t_i) / P(0, t_{i+1}) − 1) / δ
        """
        L0 = np.zeros(self.n_tenors)
        for i, t in enumerate(self.tenors):
            t_next = t + self.delta
            # 折现因子（连续复利：P(0,t) = exp(−s(t)·t)）
            s_t = self.yc.spot_rate(t) if hasattr(self.yc, 'spot_rate') else self.yc.forward_rate(t)
            s_tnext = self.yc.spot_rate(t_next) if hasattr(self.yc, 'spot_rate') else self.yc.forward_rate(t_next)
            P_t = np.exp(-s_t * t)
            P_tnext = np.exp(-s_tnext * t_next)
            L0[i] = (P_t / P_tnext - 1) / self.delta
        return L0

    def simulate(
        self,
        n_scenarios: int,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        模拟 LMM forward rate 路径（tenor 步长 Δt = δ，共 N 步）。

        返回
        ----
        L_paths   : shape (n_scenarios, n_tenors, n_tenors+1)
                    L_paths[s, i, t] = 情景 s 在时刻 t·δ 时，第 i 个 forward rate
        r_paths   : shape (n_scenarios, n_tenors+1)，短期利率近似（当期最短 live rate）
        disc_paths: shape (n_scenarios, n_tenors+1)，累计折现因子
        """
        N = self.n_tenors
        delta = self.delta
        rng = np.random.default_rng(seed)

        # L[s, i] = 当前 forward rates（初始化为 L_i(0)）
        L = np.zeros((n_scenarios, N, N + 1))
        L[:, :, 0] = self.L0[None, :]

        r_paths = np.zeros((n_scenarios, N + 1))
        disc_paths = np.zeros((n_scenarios, N + 1))
        disc_paths[:, 0] = 1.0

        # p(t)：当前时刻 t·δ 的存活最短 tenor 索引
        for t in range(N):
            p = t  # 存活 forward rates：L_{p}, L_{p+1}, ..., L_{N-1}

            # r(t) ≈ L_p(t)（最短存活 forward rate）
            r_paths[:, t] = L[:, p, t]

            # 折现：disc(t+1) = disc(t) / (1 + δ·L_p(t))
            disc_paths[:, t + 1] = disc_paths[:, t] / (1 + delta * L[:, p, t])

            if t == N - 1:
                break

            # 生成相关随机数：shape (n_scenarios, N−p)
            n_live = N - p
            Z_ind = rng.standard_normal((n_scenarios, n_live))
            Z_corr = (self.chol[p:N, p:N] @ Z_ind.T).T  # shape (n_scenarios, n_live)

            # Euler 更新每个存活 forward rate
            for j_idx, i in enumerate(range(p, N)):
                sigma_i = self.sigma_vols[i]
                L_i = L[:, i, t]

                # Spot measure drift：μ_i = Σ_{k=p}^{i} δ·σ_k·σ_i·ρ_{ki}·L_k / (1+δ·L_k)
                drift = np.zeros(n_scenarios)
                for k_idx, k in enumerate(range(p, i + 1)):
                    rho_ki = self.rho_matrix[k, i]
                    L_k = L[:, k, t]
                    drift += delta * self.sigma_vols[k] * sigma_i * rho_ki * L_k / (1 + delta * L_k)

                # Euler 步：L_i(t+1) = L_i(t) · exp((μ_i − 0.5σ_i²)·δ + σ_i·√δ·Z_i)
                log_return = (drift - 0.5 * sigma_i ** 2) * delta + sigma_i * np.sqrt(delta) * Z_corr[:, j_idx]
                L[:, i, t + 1] = np.maximum(L_i * np.exp(log_return), 1e-6)  # 保证正值

            # 已"到期"的 forward rate 不再更新（保持最后值）
            if p > 0:
                L[:, :p, t + 1] = L[:, :p, t]

        r_paths[:, N] = r_paths[:, N - 1]  # 填充最后一步

        return L, r_paths, disc_paths
