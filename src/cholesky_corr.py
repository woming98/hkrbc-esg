"""
cholesky_corr.py
================
利用 Cholesky 分解生成具有相关性结构的多维随机数。

用于将独立标准正态随机数（Z_r, Z_eq, Z_cs, Z_fx）转换为
具有指定相关性矩阵的随机数，供各风险因子模型使用。

相关性矩阵建议（基于 HK 市场历史观察）：
         IR    EQ    CS    FX
IR  [  1.00, -0.20, -0.30,  0.10 ]
EQ  [ -0.20,  1.00,  0.40, -0.15 ]
CS  [ -0.30,  0.40,  1.00, -0.10 ]
FX  [  0.10, -0.15, -0.10,  1.00 ]

IR  = 利率（interest rate shock）
EQ  = 股票（equity shock）
CS  = 信用利差（credit spread shock）
FX  = 汇率（FX shock）
"""

import numpy as np


def generate_correlated_normals(
    corr_matrix: np.ndarray,
    n_scenarios: int,
    n_steps: int,
    seed: int = 42,
) -> np.ndarray:
    """
    生成相关性随机数矩阵。

    参数
    ----
    corr_matrix : shape (n_factors, n_factors)，相关系数矩阵（必须为正定矩阵）
    n_scenarios : 情景数量
    n_steps     : 时间步数
    seed        : 随机种子

    返回
    ----
    Z_corr : shape (n_factors, n_scenarios, n_steps)
             Z_corr[i] 为第 i 个风险因子的相关随机数矩阵
    """
    np.random.seed(seed)
    n_factors = corr_matrix.shape[0]

    # Cholesky 分解：corr = L @ L.T
    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        # 若矩阵不正定，进行对角线微调（正则化）
        epsilon = 1e-6
        L = np.linalg.cholesky(corr_matrix + epsilon * np.eye(n_factors))

    # 生成独立标准正态随机数：shape (n_factors, n_scenarios * n_steps)
    Z_indep = np.random.standard_normal((n_factors, n_scenarios * n_steps))

    # Cholesky 变换引入相关性
    Z_corr_flat = L @ Z_indep  # shape (n_factors, n_scenarios * n_steps)

    # 重塑为 (n_factors, n_scenarios, n_steps)
    Z_corr = Z_corr_flat.reshape(n_factors, n_scenarios, n_steps)

    return Z_corr


def default_correlation_matrix() -> np.ndarray:
    """
    返回 HK 市场默认相关性矩阵（4 因子：IR, EQ, CS, FX）。
    可在 config/esg_config.yaml 中覆盖。
    """
    return np.array([
        [1.00, -0.20, -0.30,  0.10],
        [-0.20,  1.00,  0.40, -0.15],
        [-0.30,  0.40,  1.00, -0.10],
        [0.10, -0.15, -0.10,  1.00],
    ])
