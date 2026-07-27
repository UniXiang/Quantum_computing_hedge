# SDD Progress Ledger — quantum_hedge

## 当前状态（2026-07-27）
- **本地**：85 passed + 1 skipped，工作树干净（db6591a）
- **壁仞远端**：79 passed + 7 skipped in 17.85s，SSH 双因素已配置，代码已同步
- **下一个 Task**：T3.3（壁仞远端 n=24/28 QAOA vs SA 实验，验收指标=可行子空间内相对 gap）

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
