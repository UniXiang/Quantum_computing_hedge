# Quantum Hedge — 量子组合优化对冲策略

> **ai4sci 比赛项目** · 国产 GPU（壁仞 Biren106M）量子计算赛道  
> A 股 alpha 池 + 对冲工具 beta池 → 下行半方差 QUBO → Ising 编码 → QAOA 离散选择 → 经典权重分配 → 日频回测

---

## 架构一览

```
Alpha 池 (A股因子) ─┐
                     ├─→ 条件下行协方差 QUBO ─→ Ising 编码 ─→ QAOA ─→ 离散持仓选择 ─→ SLSQP 连续权重 ─→ 回测
Beta 池 (对冲工具)  ─┘                              ↑
                                              壁仞 Biren106M
                                              (unitarylab + torch)
```

**核心思路：** 用量子近似优化算法（QAOA）在 24~28 变量的组合空间中搜索最优离散持仓，在控制下行风险的同时实现 Beta ≈ 0.6 的对冲目标。

---

## 关键特性

| 特性 | 说明 |
|---|---|
| **量子核** | 自研 `IsingQAOA` 模块，支持 autograd / 伴随反传 / complex64 / gradient checkpointing / INTERP 热启动 |
| **QUBO 建模** | 条件下行协方差矩阵 + 可变持仓惩罚 + 多空互斥 + Beta 偏差 + 软持仓成本 + 换手约束 |
| **双轨求解** | QAOA（壁仞 GPU） + Simulated Annealing（本地 CPU）双保险，QAOA 未达全局最优时 SA 兜底 |
| **真实数据管线** | 指定 A 股 + 美股 + BTC/XAU/CL 期货数据，严格截断无未来函数 |
| **跨平台** | 本地 WSL2 开发，壁仞 Biren106M 32GB 远端执行 |
| **全链路验证** | 104 passed + 1 skipped（本地），比特序 & Ising 能量约定钉死测试 |

---

## 项目结构

```
quantum_hedge/
├── README.md                       ← 本文档
├── CLAUDE.md                       ← AI 助手项目上下文
├── configs/
│   └── portfolio_default.yaml      ← 组合默认配置
├── docs/
│   ├── progress.md                 ← 进度台账
│   └── 2026-07-25-*-design.md      ← 设计文档
├── src/
│   ├── ising_qaoa.py               ← 量子核：IsingQAOA
│   ├── qubo_builder.py             ← QUBO 构建器
│   ├── data_loader.py              ← 日线数据加载
│   ├── real_portfolio.py           ← 真实组合 pipeline
│   ├── run_real_portfolio.py       ← 18股+3合约组合 CLI
│   ├── run_real_qaoa.py            ← QAOA 入口（壁仞）
│   ├── solvers.py                  ← 经典基线（穷举 + SA）
│   └── validate_n24_n28.py         ← n=24/28 验证
├── tests/                          ← 104 + 1 测试用例
├── scripts/                        ← 远端运行 & 数据下载脚本
├── data/
│   └── crypto_daily/               ← BTC / XAU / CL 日线
├── results/                        ← 实验结果 JSON / log
└── third_party/                    ← unitarylab vendor 副本（不动源码）
```

---

## 环境要求

### 本地（WSL2 开发）

| 组件 | 版本 |
|---|---|
| Python | 3.12 |
| torch | —（本地仅 CPU 验证） |
| unitarylab | ✅ |
| pytest / numpy / scipy / sklearn | ✅ |

### 远端（壁仞 Biren106M 执行）

| 组件 | 版本 |
|---|---|
| Python | 3.10 |
| torch | 2.9.0+cu128 |
| unitarylab | 1.0.0 |
| SDK | birensupa 1.11.0.0.rc2 |

---

## 快速开始

### 本地测试

```bash
# 安装依赖
pip install pytest numpy scipy scikit-learn

# 跑全量测试
python -m pytest tests/ -v
```

### 远端执行

```bash
# 1. 同步代码到壁仞
rsync -avz --exclude='third_party' --exclude='.git' --exclude='data' \
      --exclude='.superpowers' --exclude='results' --exclude='__pycache__' \
      -e "sshpass -e ssh" \
      /mnt/f/Gaming/quantum_hedge/ biren:/workspace/quantum/quantum_hedge/

# 2. SSH 登录并执行
sshpass -e ssh biren
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
python -m pytest tests/ -v
```

### 运行真实组合

```bash
# 本地 SA 基线 + QUBO 构建
python src/run_real_portfolio.py

# 壁仞 QAOA 求解
python src/run_real_qaoa.py
```

---

## 实验进度

| Task | 内容 | 状态 |
|---|---|---|
| T1 | IsingQAOA 量子核 | ✅ |
| T2 | QUBO + SA + n=16 验证 | ✅ |
| T3.1 | e_vec 分块 + INTERP 热启动 | ✅ |
| T3.2 | complex64 + checkpoint + 壁仞适配 | ✅ |
| T3.3 | 壁仞 n=24 实验 | ✅ 命中基态 |
| T4 | 金融 pipeline（真实 n=24） | 🟡 壁仞 p=1 命中基态，SA 保留兜底 |
| T5 | 回测 | ⬜ |

> 详细进度见 [`docs/progress.md`](docs/progress.md)

---

## 关键约定

- **比特序**：qubit 0 = LSB，自旋 z = 1 − 2x（全链路自洽）
- **Ising 能量**：E(z) = Σ h_i·z_i + Σ_{i<j} J_ij·z_i·z_j（上三角计一次，J 对称零对角）
- **无未来函数**：data_loader 和 qubo_builder 不得触碰 end_date 之后的数据
- **不动 third_party/** 源码；不使用 `QAOAAlgorithm.run()`（基类入口已覆写）

---

## 团队分工

| 线 | 负责人 | 内容 |
|---|---|---|
| A | 同事 | 壁仞平台 spike 与 GPU 对接 |
| B | — | ETF / BTC / XAU / CL 数据 |
| C | — | 量子核 + QUBO + 经典基线（本仓库主体） |
| D | — | 金融 pipeline（alpha_selector） |
| E | — | 回测 |

---

## License

本项目为 ai4sci 比赛参赛作品。
