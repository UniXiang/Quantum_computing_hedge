# SDD Progress Ledger — quantum_hedge

## 当前状态（2026-07-27）
- **本地**：T3.3 实现完成，95 passed + 1 skipped；尚未提交
- **壁仞远端**：T3.2 代码 79 passed + 7 skipped in 17.85s；T3.3 尚未同步
- **当前 Task**：T3.3 壁仞 n=24 跑数完成；待决定是否用 INTERP 重跑 p=2/4
- **计划调整**：主结果固定为 n=24；n=28 不再作为验收项，仅保留为后续资源外推

Task 4.1: real n=24 classical pipeline complete; Biren QAOA pending
- 用户确认配置：持仓数量不固定；其余采用默认方案
  R1-D1-H(OKX多空)-B1-W1-T1-S1-M1
- 24变量新分配：18只股票决赛候选 + BTC/CL/XAU各long/short两个方向；
  同合约禁止同时多空
- 新增 `configs/portfolio_default.yaml`：软持仓成本、Beta目标0.6、
  周频、QAOA p=1 + SA回退、连续凸优化约束
- 新增 `benchmark_downside_covariance`：沪深300下跌日保留资产完整正负
  收益，能够用负交叉协方差识别真实对冲
- 新增 `build_flexible_selection_qubo`：无精确K惩罚，包含超额收益、
  条件下行风险、Beta偏差、软持仓成本、换手、OKX方向互斥
- 回测口径明确为 `trend_only_simplified`：long=+r、short=−r；暂不计
  手续费、滑点、资金费率、保证金、强平、合约乘数、基差/移仓。该结果
  只代表价格方向实验，不代表可实现净收益
- 验证：`tests/test_qubo_builder.py` 17 passed；全仓 104 passed + 1 skipped
- 用户指定的18只股票缓存全部存在；实际目录为 `../bs_cache_1year`
- 共同截止日 2026-07-03（受XAU数据末日限制），CL限制实际共同窗口为
  83个A股交易日，其中沪深300 ETF下跌39日
- 新增 `real_portfolio.py`：严格截断数据、BTC/CL/XAU价格级对齐、
  价格多因子、市场beta、24变量真实QUBO、SLSQP连续仓位
- 选择代理尺度为股票5%、合约方向8%，只用于Hamiltonian量纲，不是
  最终仓位或固定数量；持仓成本按代理敞口缩放
- 2026-07-03 SA结果：8只股票 + BTC/CL/XAU short，连续层总敞口
  52.90%、净敞口40.00%、beta=0.6192；24-bit全能量表确认SA选择与
  全局最优逐位一致，能量差约2e-15
- 输出：`results/real_portfolio_2026-07-03.json`；
  `results/real_n24_instance.npz` 可直接送壁仞，不需要上传私人行情缓存
- 本地全仓验证更新为104 passed + 1 skipped

Task 3.3: experiment complete (n=24; mixed QAOA result)
- 交付：`validate_n24_n28.py`（确定性合成因子收益组合实例、e_vec跨p复用、
  精确可行子空间分块枚举、QAOA top-4096 后按 K 后选择、同标称预算 SA、
  每个深度断点写 JSON/Markdown）、`t33_remote_run.sh`
- 新增逐层解析伴随反传，梯度与 eager autograd 在 complex128/64 下钉死；
  主验收规模已调整为 n=24，不再以分析模型宣称 n=28 可装入 32 GiB
- 大规模 Stage 3 改为 torch 原生设备内 `topk`，避免 unitarylab 将
  n=28 complex128 全态矢量搬回主机；小 n 默认仍走 unitarylab 锚点
- 验收指标：
  `gap=(E_candidate-E_feasible_best)/(E_feasible_worst-E_feasible_best)`，
  越低越好；非 K 解不计分，禁止用惩罚主导的全空间 worst 美化结果
- 本地验证：95 passed + 1 skipped；远端同步后 89 passed + 7 skipped；
  n=12 CPU/SUPA 伴随梯度相对差 < 1e-6。执行
  `sshpass -e ssh biren 'bash -s' < scripts/t33_remote_run.sh`
- 壁仞适配实测：
  - OpenBLAS 0.3.20 默认 64 线程会静默污染 n=24 高瘦 GEMM 后续分块；
    `_energy_vector` 已局部锁为单 BLAS 线程，n=24 全表映射复核通过
  - SUPA 不支持 n=24 高维 shape，mixer 已改为 LSB 等价的
    `(-1, 2, 2^k)` rank-3 分组；训练显存稳定约 7.6–8.6 GiB
  - p=1：QAOA 557.1s，命中 K=12 精确最优，feasible gap=0.0；
    同预算 SA 260.7s，gap=0.0
  - p=2：QAOA 1099.4s，top-4096 无 K=12 解（gap 不计）；
    SA 510.4s，gap=0.0
  - p=4：QAOA 2187.8s，top-4096 无 K=12 解（gap 不计）；
    SA 1019.5s，gap≈7.8e-11
  - 失败形态：p=2 高概率池主要为 K=9，p=4 主要为 K=19；
    随机初始化在更深电路上落入错误基数扇区，后续可用 INTERP 重跑验证
  - 结果：远端 `results/t33_n24.{json,md}`；本地副本
    `results/t33_n24.remote.{json,md}`

Task 3.2: complete (commits 2a72345..db6591a, review clean)
- 交付：complex64开关（默认complex128不变）、per-layer checkpointing（默认False）、resolve_device适配（biren/supa统一入口+清晰报错）、t32_remote_check.sh远端一键脚本、16项新测试
- 显存模型：n=24 p=4 complex64+checkpoint峰值 7.125 GiB < 8 GiB验收目标（基线complex128无checkpoint ~50.9 GiB ≫ Biren106M 32GB）；地雷#2理论关闭
- 测试：85 passed + 1 skipped (torch_br本地缺失，预期行为)
- 审查 Important(1)：docstring写~7.7GB vs estimate_evolve_memory输出7.65GB略有出入（四舍五入量级一致），可留待后续纠；Minor(4)：test_resolve_device_invalid用过于宽泛Exception、checkpoint梯度测试未按dtype参数化、远程脚本memory API缺失时静默通过、1j→complex转换使complex64路径真正可行

Task 3.1: complete (commits 99cb205..2a72345, review clean)
- 交付：分块e_vec+缓存（n=24峰值641MB<1GB，旧算法n=20即4.9GB）、_get_h_cost guard(n>12 raise)+run()隔离、best-seen（pre-step快照，审查者变异测试确认有效）、INTERP（Zhou et al. 2020公式）、集成测试test_integration.py
- n=16 solve: 186.6s→49.3s (3.8x)；68 passed（审查者本机复跑确认）
- INTERP 收敛收益待 T3.3 在 n≥24 实测；n=28 分块峰值推算~770MB裕度不大（chunk_size已参数化）
- Task 3.1 审查 Minor（待分拣）：报告分项计数写错(16实为13)；test_energy_vector_no_dense_spin_table名称夸大(实为功能冒烟)；cobyla train_level能量排序语义差异；e_vec无类型检查

## 最终整体审查（2026-07-26, opus）: 可合并 ✅ 无 Critical/Important
- 跨模块比特序/offset 全链路自洽；设计文档 4.2/4.3 全部落地
- Minor 分拣：全部"留待后续/忽略"，合并前零必修
- 设计承诺缺口（排入 n=24/28 任务）：INTERP 初始化、L-BFGS、best-seen 追踪、solve_gurobi
- n=24/28 地雷清单（按严重度）：
  1. _energy_vector (2^n,n) 中间矩阵 n=24≈37GB + 训练循环每次迭代重算 → 第一阻塞点
  2. autograd 计算图 n=24/p=4 ≈77GB 超 BR100 64GB → 需 complex64+checkpointing
  3. _get_h_cost 返回 dense (2^n,2^n)，n=16 即 68GB，公共方法误调即 OOM → 加 guard/废弃基类 run()
  4. INTERP 缺失，n≥24 barren-plateau 区域收敛风险
  5. solve() Stage3 走 unitarylab 编译执行器全态矢量，壁仞上行为未验证（依赖 A 线 spike）
- 测试缺口：跨模块集成测试（bitstring→x→x'Qx 在 pytest 内钉死）、_get_h_cost 大n guard 测试

Task 1: complete (commits e6d5784..55072a8, review clean)
Task 2: complete (commits 55072a8..99cb205, review clean after 1 fix round)
- 交付：data_loader（无未来函数截断）、qubo_builder（半方差+LW收缩+基数/换手惩罚）、solvers（穷举/SA，SA含_flip_delta一致性测试）、validate_n16（n=16 三通道验证，ratio 0.9999，报告含可行子空间景观诚实披露）
- 测试：44 passed
- 重要结论：n=16 实例基数惩罚主导景观（可行子空间能量散布仅全程0.01%），0.95阈值近乎vacuous——n=24/28 任务必须以"可行子空间内相对gap"为验收指标
- QAOA n=16 CPU 训练 ~160s → n≥24 必须 GPU（设计风险#2）
- 观察：results/ 未纳入版本控制（gitignored）；工作树有外部未跟踪文件（GPU使用对接接.md、data/、scripts/，疑为A线产出）
- 交付：src/ising_qaoa.py（IsingQAOA 加权Ising+局部场，autograd/cobyla双通道）、tests 15 passed、smoke脚本
- 比特序约定：qubit 0 = LSB，z = 1−2x（模块 docstring 锁定）
- ⚠️项已核实无阻塞：unitarylab executor 不支持角度微分（替代路径数学等价性经审查者独立验证 2e-7）；基类 kron 比特序疑为库内 bug（未动 third_party）

## Minor findings（留给最终整体审查分拣）
1. ising_qaoa.py:304 `np.random.seed(seed)` 全局RNG污染（冗余，建议删）→ Task 2 §0 已修复
2. ising_qaoa.py:260 autograd 以末次迭代能量择优，非轨迹最优
3. ising_qaoa.py:310 eigvalsh O(2^3n) 冗余（H严格对角）→ Task 2 §0 已修复（e_vec.min()）
4. ising_qaoa.py:320 energy_history 多次restart拼接未在 docstring Returns 说明
5. smoke_qaoa.py:56 exact=0 时除零

## Task 2 审查 Minor findings（2026-07-26，待最终审查分拣）
6. ising_qaoa.py:299 solve() docstring 仍写 "dense diagonalization"（§0 修复后残留旧文档）
7. validate_n16.py:93 energy_table(Q) 计算两次（n=16 无实际影响）
8. test_solvers.py:83 预算测试上界 2.0s 对 0.3s 预算过宽，慢机器 flaky 风险
9. data_loader.py:47 前缀规则误判北交所 4/8 开头为 sz（有 glob fallback 兜底）

## Task 2 ⚠️项处置（编排层已核实，不阻塞）
- "SA 初版 delta bug 是否存在过"无法回溯（单 commit 无中间历史），但审查者独立推导确认当前公式正确，且 Important #1 已要求补正式增量/全量一致性测试
- n16 运行数值未重跑验证，但收敛曲线形态合理、三通道数值与测试体系自洽
