# Quantum Hedge

跨 A 股、美股、黄金、原油和比特币的量子近似组合优化研究项目。

## 当前能力

- 1 个资产对应 1 个量子比特：`1=做多`，`0=不持有`
- 不做空，离散选择后执行连续满仓权重优化
- 固定汉明重量 Dicke 初态与环形 XY mixer
- QAOA、模拟退火和精确枚举基准
- walk-forward 跨市场回测
- top-K 候选连续权重重优化
- p=1–4、多种子与 CVaR 实验
- SA warm-start QAOA
- n=24、28、30、32 且 K=8 的规模实验
- n=32 下 K=4–9 的持仓数量搜索

## 主要目录

```text
configs/   投资组合与 QAOA 参数
docs/      设计文档和进展记录
scripts/   数据准备和远端 GPU 运行脚本
src/       QUBO、QAOA、SA、回测和报告实现
tests/     单元测试与集成测试
```

市场数据、缓存、运行结果和本地环境文件不会提交到仓库。

## 本地测试

```bash
python -m pytest -q
```

## 主要实验入口

```bash
PYTHONPATH=src python src/prepare_fixed_k_scale.py \
  --config configs/design_long_n24.yaml \
  --n 32 --K 8 --output-dir results/fixed_k_scale/n32

PYTHONPATH=src python src/run_fixed_k_warm_qaoa.py \
  --instance results/fixed_k_scale/n32/instance.npz \
  --context results/fixed_k_scale/n32/context.json \
  --output results/fixed_k_scale/n32/result.json \
  --device cpu
```

壁仞 GPU 环境使用 `--device biren`，并需事先加载对应 SUPA SDK。

## 重要说明

当前 GPU 运行属于量子线路模拟，不是真实量子硬件。SA warm-start QAOA
能够把经典候选集中为高概率量子态，但不能据此宣称量子优势。研究结论应同时
报告经典搜索成本、QAOA 调参成本、shots 命中率和精确/近似最优差距。
