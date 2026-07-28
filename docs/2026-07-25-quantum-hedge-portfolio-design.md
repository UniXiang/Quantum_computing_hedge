# 量子组合优化对冲策略 — 设计文档

- **日期**：2026-07-25
- **比赛**：ai4sci 细分赛道——国产 GPU（壁仞平台）上的量子计算
- **方案**：方案 A —— 经典预筛选 + 量子 QUBO 优化核
- **量子库**：[unitarylab_algorithms](https://github.com/unitarylab/unitarylab_algorithms.git) + [quantum-skills](https://github.com/unitarylab/quantum-skills.git)

---

## 1. 背景与目标

### 1.1 策略思想

构建日频调仓的多空对冲组合：

- **alpha 池**：约500只可交易股票 → 价格多因子/板块分散漏斗 → 18只决赛候选
- **beta 池（对冲工具）**：OKX BTC/CL/XAU 合约，每个允许 long/short/off
- **目标**：不固定入选数量；量子优化器选择股票与合约方向，经典优化器分配连续仓位，使超额收益、市场下跌日风险和 Beta=0.6 目标达到折中

### 1.2 已确认的关键决策（头脑风暴结论）

| 维度 | 决策 |
|---|---|
| 量子地位 | 比赛核心评审点，必须展示量子 vs 经典求解器的严格对比 |
| 目标函数 | 价格多因子超额收益 − 市场下跌日条件协方差风险 − Beta偏差/换手/持仓成本 |
| beta 池 | OKX BTC-USDT-SWAP、CL-USDT-SWAP、XAU-USDT，均允许多空但禁止同合约同时多空 |
| alpha 池 | 约500只 → 价格多因子 + 板块分散 → 18只决赛候选 |
| 入选数量 | 不固定；软持仓成本控制稀疏度，不使用精确 K 惩罚 |
| 调仓频率 | 周频，t日收盘信号、t+1收益生效 |
| 计算平台 | 壁仞 GPU（BR100，64GB HBM），PyTorch torch backend |
| 回测口径 | 第一版仅按价格走势计算；不含手续费、滑点、资金费率、保证金、强平、合约乘数、基差/移仓 |

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

### 2.2 交叉窗口（n=24 主结果；n=28 可选资源外推）

穷举需分钟-小时级，QAOA 在可接受时间内达到高逼近率。展示：
- 逼近率随 QAOA 层数 p 的收敛曲线
- 同时间预算下 QAOA vs 模拟退火的解质量对比

### 2.3 资源外推区（n>24，不作为当前验收）

- 当前主结果固定 n=24；更大规模仅报告理论态矢量/分析内存，不宣称已在32GB Biren106M实跑
- 经典基线使用 SA；量子结果不可行时必须回退，不以全空间惩罚项美化结果

### 2.4 量子独有能力：解分布采样

QAOA 输出是概率分布而非单解。top-K 高概率比特串 = K 个近最优组合，其交集/分散度提供持仓置信度信息——经典精确求解器只给 argmin，无法提供。做成"解分布分析"实验。

### 2.5 诚实预期（写进报告，不回避）

- QAOA 在任何规模都不宣称打败最优经典求解器
- 展示的是：方法在经典精确方法死掉的尺度上依然成立 + 通往真硬件的资源估算（物理 qubit 数、shot 数）
- 下行半方差矩阵估计噪声大，需收缩（shrinkage）处理，作为稳健性实验呈现

---

## 3. 系统架构

```
经典漏斗（约500只）              量子核（GPU, n=24）           经典权重/走势回测
可交易股票池
  ↓ 20/60/120日动量 + 下行调整收益 + 低波动 + 流动性 + 板块分散
alpha 决赛候选 18只
OKX BTC/CL/XAU × (long, short) = 6个方向变量
  ↓ 市场下跌日条件协方差 Σ↓、超额收益 α、Beta向量
QUBO 矩阵 → Ising (h, J)
  ↓ QAOA 求解（unitarylab torch backend, device=壁仞GPU）
top-K 比特串（近最优组合分布）
  ↓ 方向互斥检查；不可行则SA回退；选中子集上连续凸优化
目标持仓
  ↓ t+1价格收益（第一版忽略交易/合约现实细节）
简化走势净值 / alpha与对冲方向归因
```

**两阶段分解**：量子负责离散选择（股票进/出 + OKX方向），经典负责连续多空权重。这是 qubit 预算下的标准做法。

---

## 4. QUBO 编码设计（核心技术点）

### 4.1 qubit 分配（n=24 已确认配置）

| 段 | qubit 数 | 含义 |
|---|---|---|
| alpha 候选选择 | 18 | 每只股票 1 qubit，1=入选做多 |
| BTC方向 | 2 | long / short；允许均为0，禁止同时为1 |
| CL方向 | 2 | long / short；允许均为0，禁止同时为1 |
| XAU方向 | 2 | long / short；允许均为0，禁止同时为1 |

入选数量不固定。`holding_cost * sum(x)` 是稀疏正则，不是目标数量；同合约 long/short 用 `conflict_penalty*x_long*x_short` 互斥。

### 4.2 目标函数

```
min  v'·Σ↓·v − λR·alpha'·v + λB·(beta'·v−0.6)^2
     + holding_cost'·x + γ·|x−x_prev|
     + Σcontract Aconflict·x_long·x_short
```

- `v = sign*x` 是选择阶段的方向代理暴露；股票/long为+1，short为−1
- Σ↓：只取沪深300下跌日，但保留所有资产完整正负收益的条件协方差，做 Ledoit-Wolf 收缩
- alpha：20/60/120日价格多因子预测相对候选池平均的超额收益
- 数量由目标函数内生决定；连续仓位在入选子集上另行优化
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
| `alpha_selector.py` | 约500只股票价格多因子 + 板块分散，日产出18只决赛候选 | `select(date) -> DataFrame[code, sector, factor_score]` | bs_cache_1year，concept_dragon 复用 |
| `beta_pool.py` | BTC/CL/XAU日收益并展开long/short方向变量 | `returns(date, window) -> DataFrame` | 已有OKX日线CSV |
| `qubo_builder.py` | 滚动窗口 → Σ⁻(收缩)、μ → QUBO Q 矩阵 → Ising (h, J) | `build(alpha_df, beta_df, prev_w, date) -> (h, J, meta)` | numpy, sklearn |
| `ising_qaoa.py` | 继承 unitarylab QAOAAlgorithm，加权 Ising + 局部场 + autograd 训练 | `solve(h, J, layers, device) -> dict[bitstring, prob]` | unitarylab_algorithms |
| `solvers.py` | 三通道基线：穷举（n≤16）/ SA / Gurobi 限时（n≥30） | `solve_exact(Q)`, `solve_sa(Q, budget)`, `solve_gurobi(Q, limit)` | scipy, gurobipy(可选) |
| `weight_allocator.py` | 选中子集上的经典连续权重（半方差 QP 求解 / 风险平价） | `allocate(selected, Sigma_minus) -> w` | scipy |
| `backtest.py` | t日信号 → t+1价格方向收益；第一版不含执行/合约成本细节 | `run(signals) -> nav, attribution` | 简化走势实验 |
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

## 7. 回测设计（第一版简化走势实验）

- **信号时点**：t 日收盘后出信号，使用 t+1 日收益，严格无未来函数
- **调仓**：周频，单次目标换手上限20%
- **合约方向收益**：long使用 `+r`，short使用 `−r`
- **明确忽略**：手续费、滑点、资金费率、保证金、强平、合约乘数、基差和移仓
- **解释边界**：输出只表示价格走势上的方向策略收益，不是可实现净收益或实盘风险报告
- **评估指标**：年化、最大回撤、下行偏差、Calmar、对冲腿贡献归因（alpha 收益 vs 对冲损益分解）
- **replay 验证**：按用户 10 步复盘框架执行关键日回放
- **禁令遵守**：无未来函数、信号 repaint 检查（同一日期两次运行结果一致）

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
2. **M1（第 2-3 周）**：Ising 扩展 + 三通道基线 + n=16/24 逼近率曲线（n=28 仅作可选资源外推）
3. **M2（第 3-4 周）**：18只股票+3个OKX多空方向的真实n=24组合 + 连续权重 + 简化走势回测
4. **M3（第 4-5 周）**：稳健性实验（窗口/收缩/layers）+ 解分布分析 + 报告撰写
