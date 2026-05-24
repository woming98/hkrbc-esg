# HKRBC ESG Generator

**香港 RBC 分红基金 TVOG 专用经济情景生成器（Python 开源实现）**

> 面向中小型香港保险公司，提供符合 Cap. 41R Rule 19 要求的风险中性（Q-measure）经济情景，
> 用于计算分红基金（Par Fund）的 **TVOG（Time Value of Options and Guarantees）**，
> 无需购置 Moody's / Conning / FIS 等商业 ESG 软件授权。

---

## 监管依据

| 要求 | 条款 | 本项目实现 |
|---|---|---|
| ≥1,000 条情景 | Cap. 41R Rule 19 | 默认 1,000 条，可配置 |
| 市场一致性 | Rule 19：market-consistent | HW1F 精确拟合 IA Schedule 4 yield curve |
| 无套利 | Rule 19：arbitrage-free | Martingale Test 自动验证 |
| 无风险折现率 | Schedule 4 | 直接读取 IA 发布的零息利率 |
| UFR | IA G.N. 4006 of 2024 | 配置文件中指定（HKD/USD: 3.8%） |

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 更新 IA Schedule 4 利率曲线（每季度）
# 编辑 config/esg_config.yaml 中的 ia_schedule4.spot_rates

# 3. 生成情景
python examples/run_esg.py

# 4. 指定情景数量
python examples/run_esg.py --scenarios 2000

# 输出文件
# output/esg_scenarios.csv     ← 导入 Excel / Python 计算 TVOG
# output/esg_scenarios.ESC     ← 导入 FIS Prophet
# output/esg_summary.csv       ← 情景统计摘要（监管文件用）
```

---

## 项目结构

```
hkrbc-esg/
├── src/
│   ├── yield_curve.py       # IA Schedule 4 利率曲线 + 瞬时远期利率 f(0,t)
│   ├── hw1f.py              # Hull-White 1-Factor 利率模型（Q-measure 精确离散化）
│   ├── equity.py            # GBM 股票总回报模型（Q-measure）
│   ├── cholesky_corr.py     # Cholesky 相关性随机数生成
│   ├── martingale_test.py   # Martingale Test（Rule 19 合规验证）
│   └── scenario_output.py   # CSV / Prophet .ESC 格式导出
├── config/
│   └── esg_config.yaml      # 所有参数配置（利率曲线、模型参数、相关性等）
├── examples/
│   └── run_esg.py           # 主入口
├── output/                  # 自动创建，存放生成的情景文件
└── requirements.txt
```

---

## 模型说明

### 利率模型：Hull-White 1-Factor（Q-measure）

```
dr(t) = [θ(t) - a·r(t)] dt + σ·dW^Q(t)

θ(t) 由 IA Schedule 4 初始 yield curve 唯一确定：
θ(t) = ∂f(0,t)/∂t + a·f(0,t) + σ²/(2a)·(1 - e^{-2at})
```

- **优点**：解析公式（无迭代）、精确拟合初始 yield curve、Martingale Test 通过率高
- **参数**：`a`（均值回归速度）、`σ`（波动率），建议按最新 HKD swaption vol 重新校准

### 股票模型：GBM（Q-measure）

```
dS(t) = r(t)·S(t) dt + σ_eq·S(t) dW^Q_eq(t)
```

- drift = r(t)，确保无套利（Q-measure 下股票期望回报 = 无风险利率）

### Martingale Test

```python
# 债券鞅检验：
E^Q[exp(-∫₀ᵀ r(t) dt)] ≈ P(0, T)   # 允许误差 ±0.5%

# 股票鞅检验：
E^Q[S(T)/S(0) · exp(-∫₀ᵀ r(t) dt)] ≈ 1.0
```

---

## 配置参数说明

编辑 `config/esg_config.yaml`：

```yaml
ia_schedule4:
  spot_rates: [0.0335, 0.0340, ...]   # 每季从 IA 官网更新

hull_white_1f:
  a: 0.05       # 均值回归速度（建议 0.01–0.20）
  sigma: 0.010  # 利率波动率（建议 0.005–0.020）

simulation:
  n_scenarios: 1000   # HKRBC 最低 1,000 条
```

---

## Par Fund TVOG 计算流程

```
IA Schedule 4 利率曲线
        ↓
   HW1F 模拟 1,000 条利率路径
        ↓
   Martingale Test 验证（Rule 19）
        ↓
   导出情景至 Prophet / Excel
        ↓
   Prophet 按情景计算 BEL（AS 递推 + GSV floor）
        ↓
   TVOG = 随机均值 BEL − 确定性 BEL
```

---

## 适用场景

- **Par Fund TVOG**：GSV put option + RB ratchet 的随机估值
- **IFRS 17 MRB**：市场风险利益（Market Risk Benefit）的随机估值
- **ALM 分析**：Real-World 参数下的资产负债缺口压力测试（切换配置参数即可）

---

## 局限性与注意事项

1. **HW1F 不捕捉利率微笑（smile）**：对强烈的 swaption skew 建议升级至 HW2F 或 LMM
2. **参数需定期校准**：至少每季按最新 IA Schedule 4 曲线更新 `spot_rates`；`a`, `σ` 建议每年校准一次
3. **信用利差模型为简化版**：实际项目建议加入跳跃成分（Jump-Diffusion）
4. **监管沟通**：首次使用前建议与公司精算函数（Appointed Actuary）确认模型满足内部模型验证要求

---

## 参考资料

- [Cap. 41R Insurance (Valuation and Capital) Rules](https://www.elegislation.gov.hk/hk/cap!41R)
- [IA HKRBC 资源页](https://www.ia.org.hk/en/legislative_framework/hkrbc/index.html)
- [IA UFR 宪报通告 G.N. 4006 of 2024](https://www.gld.gov.hk/egazette/english/gazette/volume.do?year=2024&volume=28&number=24&part=A)
- GL34 分红基金管理指引（2024年7月生效）

---

## 免责声明

本项目为学习和参考用途。使用本工具产生的情景须经公司精算函数验证，确认符合 HKRBC 监管要求后方可用于正式估值。
