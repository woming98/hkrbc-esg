"""
HKRBC ESG Generator
====================
香港 RBC（HKRBC）分红基金 TVOG 专用经济情景生成器（Python 开源实现）。

主要模块：
- yield_curve      : IA Schedule 4 利率曲线构建及远期利率提取
- hw1f             : Hull-White 1-Factor 利率模型（Q-measure）
- equity           : GBM 股票总回报模型（Q-measure）
- cholesky_corr    : Cholesky 相关性随机数生成
- martingale_test  : HKRBC Rule 19 鞅检验（市场一致性验证）
- scenario_output  : CSV / Prophet .ESC 格式导出
"""
