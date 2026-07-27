# 量子组合优化对冲策略 — 设计文档

- **日期**：2026-07-25
- **比赛**：ai4sci 细分赛道——国产 GPU（壁仞平台）上的量子计算
- **方案**：方案 A —— 经典预筛选 + 量子 QUBO 优化核
- **量子库**：[unitarylab_algorithms](https://github.com/unitarylab/unitarylab_algorithms.git) + [quantum-skills](https://github.com/unitarylab/quantum-skills.git)

---

## 1. 背景与目标

### 1.1 策略思想

构建日频调仓的多空对冲组合：

- **alpha 池**：现有框架识别领先板块 → 板块内动量排序取 top N 个股
- **beta 池（对冲工具）**：BTC/XAU 永续（可做空）+ 场内 ETF（黄金 518880、原油、豆粕等，只能做多）
- **目标**：不预设固定对冲比，由量子优化器在下行半方差目标下自行学出非对称的权重结构（下行保护优先于上行参与）

### 1.2 已确认的关键决策（头脑风暴结论）

| 维度 | 决策 |
|---|---|
| 量子地位 | 比赛核心评审点，必须展示量子 vs 经典求解器的严格对比 |
| 目标函数 | 下行半方差（downside semivariance），非对称权重交给优化器 |
| beta 池 | BTC/XAU 永续（可做空）+ 场内 ETF（做多低/负相关） |
| alpha 池 | 板块识别 + 板块内动量排序 |
| 调仓频率 | 日频 |
| 计算平台 | 壁仞 GPU（BR100，64GB HBM），PyTorch torch backend |

### 1.3 已排除的思路（及原因）

- **静态负相关个股做对冲**：A 股个股不可做空，且事后筛选的反相关性样本外不稳定（过拟合）
- **人工指定非对称对冲比（涨对冲3%/跌对冲5%）**：静态线性资产无法表达非对称结构，只能由期权或优化器内生实现
- **股指期货 IF/IC/IM**：用户数据底座无期货数据源，且用户未选
- **端到端全量子（方案 B）**：金融逻辑被稀释，评审会质疑筛选环节量子的必要性
- **D-Wave 退火（方案 C）**：外部云依赖、不可复现，违反 reproducibility 原则

---

## 2. 量子优势的诚实叙事（比赛故事线）

NISQ 时代量子优化在小规模上不可能赢经典精确方法，本项目不宣称这一点。故事分三层：

### 2.1 正确性锚点（n=16）

16 qubit = 65536 组合，穷举毫秒级出精确解。用于：
- 验证 QUBO→Ising 编码正确性（QAOA 解 vs 精确解的逼近率）
- 快速回归测试

### 2.2 交叉窗口（n=24-28）

穷举需分钟-小时级，QAOA 在可接受时间内达到高逼近率。展示：
- 逼近率随 QAOA 层数 p 的收敛曲线
- 同时间预算下 QAOA vs 模拟退火的解质量对比

### 2.3 经典精确失效区（n=30-32，壁仞 GPU 主场）

- 2^30 ≈ 10 亿组合，穷举实际不可行；基线改为最强经典启发式（长预算 SA / Gurobi 限时）
- **CPU vs 壁仞 GPU scaling benchmark**（n=16→32 同负载）是命中"国产 GPU"赛题的核心交付物
- statevector 内存：complex128 每振幅 16B，n=30→16GB，n=31→32GB，n=32→64GB（需 complex64 压到 32GB）

### 2.4 量子独有能力：解分布采样

QAOA 输出是概率分布而非单解。top-K 高概率比特串 = K 个近最优组合，其交集/分散度提供持仓置信度信息——经典精确求解器只给 argmin，无法提供。做成"解分布分析"实验。

### 2.5 诚实预期（写进报告，不回避）

- QAOA 在任何规模都不宣称打败最优经典求解器
- 展示的是：方法在经典精确方法死掉的尺度上依然成立 + 通往真硬件的资源估算（物理 qubit 数、shot 数）
- 下行半方差矩阵估计噪声大，需收缩（shrinkage）处理，作为稳健性实验呈现

---

## 3. 系统架构

```
经典漏斗（全量）              量子核（GPU, n=30-32）          经典执行/回测
5199只股票 (bs_cache_1year)
  ↓ 板块识别（concept_dragon 复用）
领先板块
  ↓ 板块内动量排序
alpha 候选 12 只
beta 工具池 8 个 (BTC/XAU 永续 + 场内ETF)
  ↓ 滚动60日窗口 → 下行半方差矩阵 Σ⁻、收益向量 μ
QUBO 矩阵 → Ising (h, J)
  ↓ QAOA 求解（unitarylab torch backend, device=壁仞GPU）
top-K 比特串（近最优组合分布）
  ↓ 经典连续权重分配（选中子集上闭式半方差最小化）
目标持仓
  ↓ t+1 成交模拟（T+1、涨跌停、手续费、滑点）
回测净值 / 绩效归因 / replay 验证
```

**两阶段分解**：量子负责离散选择（资产进/出 + 对冲比档位），经典负责连续权重。这是 qubit 预算下的标准做法。

---

## 4. QUBO 编码设计（核心技术点）

### 4.1 qubit 分配（n=32 目标配置）

| 段 | qubit 数 | 含义 |
|---|---|---|
| alpha 候选选择 | 12 | 每只股票 1 qubit，进/出 |
| beta 工具选择 | 8 | 每个工具 1 qubit，进/出；BTC/XAU 腿允许负权重（做空） |
| 对冲比率 | 6 | 64 档，0%~150% 名义对冲比 |
| 板块间配置 | 4 | 2 个领先板块间的资金分配档位（16 档） |
| 预留 | 2 | 实验余量 |

n=16 回归配置：alpha 6 + beta 4 + 对冲比 4 + 板块配置 2。

### 4.2 目标函数

```
min  w'·Σ⁻·w  −  λ·μ'·w  +  γ·|w − w_prev|（换手惩罚）
s.t. 持仓数量约束（cardinality，作为惩罚项）
     个股权重上下限
     beta 做空腿仅 BTC/XAU 永续允许
```

- Σ⁻：滚动 60 日窗口的下行半方差协方差矩阵（只用 t 日前数据，无未来函数），做 Ledoit-Wolf 收缩
- w 由选择比特 + 档位比特解码为离散权重网格
- QUBO 矩阵 Q 经标准变换 x∈{0,1} → z∈{−1,+1} 映射为 Ising 模型：局部场 h_i + 耦合 J_ij

### 4.3 unitarylab QAOA 扩展（比赛贡献点之一）

现有 `QAOAAlgorithm`（`quantum_machine_learning/qaoa/algorithm.py`）是 MaxCut 专用：
- `_get_h_cost` 只支持等权 ZZ 边项，无局部场
- `_build_circuit` 只实现等权 ZZ 演化

扩展方式——继承 `QAOAAlgorithm` 重写两个方法：
- `_get_h_cost(h, J)`：支持加权耦合 J_ij·Z_i·Z_j + 局部场 h_i·Z_i
- `_build_circuit`：加权 ZZ 用 `cx-rz(2γJ_ij)-cx`，局部场用单比特 `rz(2γh_i)`

保持 `BaseAlgorithm` 的日志/结果导出/`parameters.json` 规范不变，与库的其他算法风格一致。

### 4.4 训练：reverse-mode 梯度（GPU 优势点）

n=30 时 COBYLA 无梯度优化每次迭代都要演化 10 亿维态矢量，不可行。改用：
- torch backend 自动微分（反向传播穿过整条 QAOA 线路），梯度代价 ≈ 2× 前向
- 优化器：Adam/L-BFGS；参数初始化用 INTERP 策略（p 层最优参数插值初始化 p+1 层）
- 参考 quantum-skills `algorithms/gradients/reverse` 指南
- COBYLA 保留为 n≤20 小规模的交叉验证

---

## 5. 组件设计

| 模块 | 职责 | 关键接口 | 依赖 |
|---|---|---|---|
| `alpha_selector.py` | 板块识别 + 板块内动量排序，日产出 12 只 alpha 候选（含所属板块标签） | `select(date) -> DataFrame[code, sector, momentum_score]` | bs_cache_1year，concept_dragon 复用 |
| `beta_pool.py` | 固定 8 个对冲工具的日收益率序列（对齐 A 股交易日历） | `returns(date, window) -> DataFrame` | baostock（ETF）+ OKX 下载（BTC/XAU） |
| `qubo_builder.py` | 滚动窗口 → Σ⁻(收缩)、μ → QUBO Q 矩阵 → Ising (h, J) | `build(alpha_df, beta_df, prev_w, date) -> (h, J, meta)` | numpy, sklearn |
| `ising_qaoa.py` | 继承 unitarylab QAOAAlgorithm，加权 Ising + 局部场 + autograd 训练 | `solve(h, J, layers, device) -> dict[bitstring, prob]` | unitarylab_algorithms |
| `solvers.py` | 三通道基线：穷举（n≤16）/ SA / Gurobi 限时（n≥30） | `solve_exact(Q)`, `solve_sa(Q, budget)`, `solve_gurobi(Q, limit)` | scipy, gurobipy(可选) |
| `weight_allocator.py` | 选中子集上的经典连续权重（半方差 QP 求解 / 风险平价） | `allocate(selected, Sigma_minus) -> w` | scipy |
| `backtest.py` | t 日信号 → t+1 成交；T+1、涨跌停、手续费、滑点；日频 replay | `run(signals) -> nav, attribution` | 复用现有回测基座 |
| `benchmark_gpu.py` | CPU vs 壁仞 GPU scaling benchmark（n=16→32） | `sweep(ns, devices) -> timing_df` | torch |
| `run_daily.py` | 日频编排：数据更新 → alpha 选择 → QUBO → 求解 → 权重 → 落盘 | CLI | 以上全部 |

### 错误处理

- 板块识别当日无领先板块（情绪冰点）→ alpha 池为空 → 当日信号 = 纯对冲腿或空仓，记录日志不报错
- 候选股当日涨停（买不进）→ 回测执行层剔除该腿，权重在剩余腿间归一，记录
- QAOA 训练不收敛（能量历史震荡）→ 回退 p-1 层参数 + SA 兜底，两解取目标函数优者，记录事件
- beta 数据缺日（crypto 与 A 股日历不齐）→ 前向填充（只用过去数据），连续缺 5 日以上该工具当日禁用
- GPU OOM（n=32 complex128）→ 自动降 complex64，日志记录精度模式

---

## 6. 数据需求与缺口

| 数据 | 现状 | 行动 |
|---|---|---|
| A 股日线 1 年 | ✅ bs_cache_1year（5199 只 pkl） | 直接用，复用 `load_1year_data.py` 加载方式 |
| 板块/概念归属 | ✅ concept_dragon.py + pywencai_concept_cache | 复用 |
| 场内 ETF 日线（518880 黄金、原油 ETF、159985 豆粕等） | ❌ 缺 | baostock 下载，与股票缓存同格式 |
| BTC/XAU 日线 1 年 | ⚠️ 现缓存多为分钟级 | 改 `download_xau_klines.py` timeframe=1d 拉取；BTC 同理 |
| 交易日对齐 | — | crypto 24/7 → 北京时间 8:00 对齐 A 股日收盘，缺失前向填充 |

---

## 7. 回测设计

- **信号时点**：t 日收盘后出信号，t+1 日开盘价成交（不假设收盘价成交）
- **约束**：A 股 T+1；涨停不买、跌停不卖；ETF 同股票规则；BTC/XAU 永续无涨跌停但有资金费率（按历史均值计入成本）
- **成本**：股票双边佣金+印花税，ETF 佣金，永续 taker 费率 + 资金费率；滑点按日波动率比例计
- **评估指标**：年化、最大回撤、下行偏差、Calmar、对冲腿贡献归因（alpha 收益 vs 对冲损益分解）
- **replay 验证**：按用户 10 步复盘框架执行关键日回放
- **禁令遵守**：无未来函数、不假设完美成交、信号 repaint 检查（同一日期两次运行结果一致）

---

## 8. 验证与测试

| 层 | 测试 |
|---|---|
| 编码正确性 | n=16 上 QAOA 最优解 vs 穷举精确解，逼近率 ≥ 0.95（p≥4） |
| Ising 扩展正确性 | 随机 (h, J) 上，扩展后 `_get_h_cost` 的本征值 vs 直接构造的 QUBO 能量一致 |
| 无未来函数 | 任取回测日中点截断数据重跑，截断日前的信号逐位一致 |
| 回测确定性 | 同一输入两次运行，成交记录完全一致 |
| 数据缺口处理 | 构造缺日/停牌样例，验证前向填充与禁用逻辑 |
| GPU 正确性 | device=CPU vs device=GPU 同参数 QAOA，最终能量差 < 1e-6（complex64 时 < 1e-3） |

---

## 9. 风险清单（按优先级）

| # | 风险 | 等级 | 缓解 |
|---|---|---|---|
| 1 | **PyTorch 无法在壁仞 GPU 运行**（BIRENSUPA 适配未知） | 🔴 阻塞级 | Spike #1：第一天跑通 `import torch; x.cuda()` + unitarylab 最小 QAOA on GPU；不通则退 PaddlePaddle 自写模拟核（工作量 +2 周）或申请平台方支持 |
| 2 | n=30 QAOA 训练时长超预期 | 🟡 | reverse-mode 梯度 + INTERP 初始化 + layers≤6；超预算则主结果落 n=28，32 只做 scaling 演示 |
| 3 | 下行半方差矩阵估计噪声 → 权重不稳定 | 🟡 | Ledoit-Wolf 收缩 + 换手惩罚 γ 调参；稳健性实验呈现 |
| 4 | 60 日窗口内板块切换 → 半方差结构失真 | 🟡 | 敏感性实验：窗口 40/60/120 对比 |
| 5 | Gurobi 无 license | 🟢 | 退长预算 SA + 次优界（relaxation bound） |
| 6 | 对冲腿与 A 股 alpha 相关性过低，"对冲"效果弱 | 🟡 | 报告诚实呈现归因分解；这本身是研究发现 |

---

## 10. 团队分工建议（对应可行性验证）

| 线 | 任务 | 验证什么 |
|---|---|---|
| A. 平台 spike（最高优先） | 壁仞环境装 uv + unitarylab，跑通 torch GPU QAOA demo；测 n=28/30/32 内存与速度 | 风险 #1、#2 |
| B. 数据 | ETF 日线 baostock 下载；BTC/XAU 日线 OKX 拉取；交易日对齐模块 | 数据缺口 |
| C. QUBO/Ising 扩展 | 继承 QAOAAlgorithm 实现加权 Ising + 局部场；n=16 穷举验证编码正确性 | 核心技术点 |
| D. 金融 pipeline | alpha_selector 复用 concept_dragon；qubo_builder + weight_allocator | 策略前端 |
| E. 回测 | backtest.py 对接现有回测基座；成本模型 | 策略后端 |

依赖关系：C、D 可并行；E 依赖 D；整合依赖 A 的结论（决定 device 路径）。

---

## 11. 里程碑

1. **M0（第 1 周）**：平台 spike 结论 + 数据补齐 + n=16 端到端跑通（CPU）
2. **M1（第 2-3 周）**：Ising 扩展 + 三通道基线 + n=16/24/28 逼近率曲线
3. **M2（第 3-4 周）**：壁仞 GPU n=30-32 + CPU/GPU benchmark + 日频回测全量 replay
4. **M3（第 4-5 周）**：稳健性实验（窗口/收缩/layers）+ 解分布分析 + 报告撰写
