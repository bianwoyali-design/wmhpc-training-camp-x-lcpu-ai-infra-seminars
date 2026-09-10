# CHUNK 衰减数值范围实验

在仓库根目录运行：

```bash
assignment02/.venv/bin/python assignment02/team/c1_flashkda/experiments/chunk_range/run.py
```

依赖当前虚拟环境的 torch 和 matplotlib。默认在 CPU 上运行，不需要申请 GPU。
结果位于 `results/`；可用 `--out /tmp/chunk-range-results` 指定其他目录。

## 设计

- CHUNK：16、32、64。每组使用同一条长度 64 输入的前缀。
- 输入是自然对数单位的**激活后 gate**，不是原始 logits。
- 常数输入：0、-0.1、-1、-3、-5，各一次；随机输入：均匀 [-5,0] 和
  10% 概率 -5 / 其余 -0.01 的混合输入，各用种子 0–4。
- 共 15 条输入 × 3 个 CHUNK × 4 条数值路径 × 5 个观察量 = 900 行统计。
- 路径：FP64；FP32；FP32 指数转存 BF16；以 FP32 顺序累加 log2 单位 gate、
  exp2 后模拟 FP32 输出 FTZ 并转存 BF16。
- 观察量：前缀和、正衰减、逆衰减，以及所有 j<=i 的直接/分解因果衰减。
  两种衰减表达式都与 FP64 的 exp(s_i-s_j) 比较。
- BF16 因子乘法在 FP32 中进行，以分离因子存储精度和乘法误差。

`metrics.csv` 保存异常计数、有限值范围与误差；`traces.csv` 保存长度 64 的
每 token 数据；`metadata.json` 保存环境与配置。矩阵的 flat index 按下三角
行优先展开，向量另提供从 1 开始的 token 位置。只把参考非零而结果为零计作
lost_nonzero，避免把 gate=0 的合法前缀和误报为异常。
存在 Inf/NaN 时误差字段留空，不通过丢弃异常元素产生貌似正常的误差。

## 本次结果

在 g=-5 的输入上，首次异常 token（从 1 开始）为：

| 路径 | 正衰减首次为零 | 逆衰减首次 Inf |
|---|---:|---:|
| FP64 | 64 内没有 | 64 内没有 |
| FP32 | 21 | 18 |
| FP32 → BF16 | 19 | 18 |
| FTZ 输出模型 → BF16 | 18 | 18 |

FTZ 模型中，CHUNK=16 的正、逆衰减均有限且非零；CHUNK=32 出现 15 个零衰减、
15 个 Inf 逆衰减；CHUNK=64 两者各 47 个。分解的因果衰减分别出现 120 和
1128 个 NaN。直接表达式没有这些 NaN，但远距离衰减仍可能下溢。

边界推算也支持这一现象：g=-5 时，末尾自然对数前缀和是 -5C。
C=16 时 exp(-80) 约 1.80e-35、exp(80) 约 5.54e34；到第 18 个 token，
exp(-90) 约 8.19e-40，低于 FP32 最小正规数，而 exp(90) 约 1.22e39，
超过 FP32/BF16 的有限范围。仅观察正衰减变零会漏掉逆衰减更早溢出的情况。

这验证了小 CHUNK 在强负 gate 下避免中间量范围问题的理由，不能证明 16
总是最优，也不能证明所有 CHUNK=32/64 输入都会失败。温和 gate 是必要对照。

## 与实际 FlashKDA 的关系及限制

源码位置：`../../FlashKDA/csrc/smxx/fwd_kernel1.cuh` 的 gate 顺序累加，
以及约 452/465 行的 `ex2_approx_ftz_f32(±g)` 转 BF16。
设计文档：`../../FlashKDA/docs/20260420-flashkda-v1-deep-dive.md`。

本实验直接控制激活后 gate，未模拟 tanh 近似激活、原始 BF16 logits、
真实 PTX exp2 近似误差、MMA 或后续 q/k 运算。FTZ 路径只是 CPU 输出冲零模型，
不是 CUDA 指令测试。直接/分解比值用于解释中间范围问题，不是完整 kernel 实现。
尚未测量最终 KDA 输出/状态误差、rescale、大 CHUNK kernel 或性能。

验证包含所有 FP64 直接/分解比值的一致性检查，以及所有数值路径 gate=0 时
比值严格为 1 的检查。脚本成功运行，两个 PNG 经目视检查；同时提供 SVG。

图表：`decay.png` 显示 64 个 token 的 log10 衰减与首次冲零位置；
`ratio_error.png` 对照 g=-1/-5 的三种 CHUNK，非有限结果明确标注，未当成零误差。
