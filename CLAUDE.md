# Quantum Hedge — 量子组合优化对冲策略

ai4sci 比赛项目（国产 GPU / 壁仞平台量子计算赛道）。

## 架构一句话

A 股 alpha 池 + 对冲工具 beta 池 → 下行半方差 QUBO → Ising 编码 → QAOA 离散选择 → 经典权重分配 → 日频回测。

## 工作目录

本地：`/mnt/f/Gaming/quantum_hedge/`
远端（壁仞 Biren106M 32GB）：`/workspace/quantum/quantum_hedge/`
设计文档：`docs/2026-07-25-quantum-hedge-portfolio-design.md`

## 环境

| 位置 | Python | torch | unitarylab | 特殊 |
|---|---|---|---|---|
| 本地 WSL2 | 3.12 (`/mnt/f/quant/quant`) | — | ✅ | venv 已装 pytest/numpy/scipy/sklearn |
| 壁仞远端 | 3.10 (`/usr/bin/python3`) | 2.9.0+cu128 | 1.0.0 | 加载环境：`source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh` |

## SSH 远端连接

```bash
# 免密用别名（需先 export SSHPASS，服务器要求 publickey+password 双因素）：
export SSHPASS='平台密码'
sshpass -e ssh biren

# ~/.ssh/config 已配好 Host biren（端口30222，IdentityFile ~/.ssh/biren_ed25519_nopass）
```

## 代码同步

```bash
# 本地→远端（排除 third_party/ .git/ data/ results/）
export SSHPASS='平台密码'
rsync -avz --exclude='third_party' --exclude='.git' --exclude='data' \
      --exclude='.superpowers' --exclude='results' --exclude='__pycache__' \
      -e "sshpass -e ssh" \
      /mnt/f/Gaming/quantum_hedge/ biren:/workspace/quantum/quantum_hedge/

# 远端运行测试
sshpass -e ssh biren "cd /workspace/quantum/quantum_hedge && \
  source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh >/dev/null 2>&1 && \
  /usr/bin/python3 -m pytest tests/ -v"
```

## 项目结构

```
quantum_hedge/
├── CLAUDE.md              ← 这个文件（新会话从这里开始）
├── docs/
│   ├── progress.md         ← 进度台账（看板）
│   └── 2026-07-25-...md    ← 设计文档
├── configs/
│   └── portfolio_default.yaml ← 用户确认的真实组合默认配置
├── src/
│   ├── ising_qaoa.py       ← 量子核：IsingQAOA（autograd/伴随反传/complex64/checkpoint/INTERP）
│   ├── qubo_builder.py     ← QUBO：条件下行协方差+可变持仓+多空互斥+Ising
│   ├── data_loader.py      ← bs_cache日线加载（end_date严格截断）
│   ├── real_portfolio.py   ← 真实24变量数据/因子/QUBO/连续仓位pipeline
│   ├── run_real_portfolio.py ← 18股+3合约SA组合CLI
│   ├── run_real_qaoa.py    ← 导出Hamiltonian的壁仞p=1 QAOA入口
│   ├── solvers.py          ← 经典基线：穷举+SA(_flip_delta验证)
│   ├── validate_n16.py     ← n=16三通道验证脚本
│   ├── validate_n24_n28.py ← T3.3 壁仞n=24 QAOA vs同预算SA实验
│   └── smoke_qaoa.py
├── tests/                  ← 104 passed + 1 skipped（本地）
├── scripts/
│   ├── t32_remote_check.sh ← T3.2 远端一键验证脚本
│   └── t33_remote_run.sh   ← T3.3 n=24断点式实验脚本
├── data/crypto_daily/      ← B线产出：BTC/XAU/CL日线（未入库）
└── third_party/            ← unitarylab_algorithms + quantum-skills vendor副本

## 关键约定（跨模块红线）

- 比特序：**qubit 0 = LSB**，自旋 **z = 1−2x**（模块 docstring 锁定，全链路自洽）
- Ising 能量：E(z) = Σ h_i·z_i + Σ_{i<j} J_ij·z_i·z_j（上三角计一次，J 对称零对角）
- 无未来函数：data_loader 和 qubo_builder 不得触碰 end_date 之后的数据
- 不动 third_party/ 源码；不使用 QAOAAlgorithm.run()（基类入口已被覆写 raise NotImplementedError）
- unitarylab executor 不支持角度微分，autograd 路径用模块内 torch 演化（等价性钉死测试 2e-7）

## 已确认的 unittestarylab 库内 bug（备忘）

基类 kron 构造与其自身执行器比特序不一致。我们模块内统一 LSB 约定绕开，未动 third_party。选手可向 unitarylab 提交 issue。

## 团队分工

| 线 | 负责人 | 内容 |
|---|---|---|
| A | 同事 | 壁仞平台 spike（GPU使用对接.md） |
| B | 你 | ETF/BTC/XAU/CL 数据（data/ 目录） |
| C | 你 + 子agent | 量子核 + QUBO + 经典基线（本仓库主体） |
| D | — | 金融 pipeline（alpha_selector，待做） |
| E | — | 回测（待做） |

## 进度看板

详见 `docs/progress.md`（仓库内，可 git 同步）。简要：

| Task | 内容 | 状态 |
|---|---|---|
| T1 | IsingQAOA 量子核 | ✅ |
| T2 | QUBO + SA + n16 | ✅ |
| T3.1 | e_vec分块 + INTERP | ✅ |
| T3.2 | complex64 + checkpoint + biren适配 | ✅ |
| T3.3 | 壁仞 n=24 实验 | ✅ 跑数完成（p=1最优，p=2/4不可行） |
| T4 | 金融 pipeline | 🟡 混合A/美股真实n=24；壁仞改进p=1命中基态，SA保留兜底 |
| T5 | 回测 | ⬜ |
