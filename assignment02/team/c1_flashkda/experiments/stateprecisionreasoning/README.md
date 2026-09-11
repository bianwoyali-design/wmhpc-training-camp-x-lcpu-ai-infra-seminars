# 讨论点 5：BF16 状态精度

[分析](ANALYSIS.md) · [结果汇总](results/SUMMARY.md)

验证普通输入、长记忆和小更新下的输出/状态误差；区分官方整体数值误差、
仅由状态舍入造成的误差，以及 API 的状态 dtype。包含长序列、流式续算和变长序列。

## 复现

需要 CUDA PyTorch、已安装的 FlashKDA、einops 和 B300。从仓库根目录运行：

```bash
bench=assignment02/team/c1_flashkda/experiments/stateprecisionreasoning
out="$bench/runs/$(date -u +%Y%m%dT%H%M%S)"
srun -p gpu -G 1 --time=00:40:00 bash -c '
  assignment02/.venv/bin/python "$1/run.py" reproduce --out "$2" &&
  assignment02/.venv/bin/python "$1/probe.py" --out "$2/probe"
' bash "$bench" "$out"
```

默认结果写入 `runs/<时间戳>/`；`--out` 指定一个不存在的新目录。
`--quick` 只测 nominal/retention 的 256 token，用于检查流程。

```bash
# CPU：对提供的 naive 实现校验 FP64 参考
assignment02/.venv/bin/python \
  assignment02/team/c1_flashkda/experiments/stateprecisionreasoning/run.py check

# 只从 CSV 重建汇总，不需要 GPU
assignment02/.venv/bin/python \
  assignment02/team/c1_flashkda/experiments/stateprecisionreasoning/run.py report \
  --out assignment02/team/c1_flashkda/experiments/stateprecisionreasoning/results
```

参考运行在 CPU，使用一条线程；GPU 只运行官方扩展。本实验比较数值，不采性能，
不需要 NCU 权限，也不依赖 GPU 频率。长序列的 FP64 临时张量需要数 GB 主存。

## 实验矩阵

D=128、CHUNK=16。普通组将 seed=0/1/2 放到三个独立 head 中；每个 seed 单独报告，
不能把三个 head 当作重复计时样本。retention 是确定性反例，只用一个 head。
所有 raw q/k/v/g/beta 先量化为 BF16，A_log=0、dt_bias=0、lower_bound=-5。

| pattern | T | 构造与用途 |
|---|---|---|
| nominal | 256、8192、65536 | 正态 q/k/v/g/beta，零初始状态，短记忆基线 |
| weak | 256、8192、65536 | g logits=-8，随机 beta；较弱衰减，非零状态 |
| near_identity | 256、8192、65536 | g=-16、beta logits=-4；状态变化更慢 |
| nonzero | 8192 | nominal 的非零初始状态对照 |
| correlated | 8192 | q/k 接近同一向量，beta logits=4，g=-8；高度相关更新 |
| dynamic | 8192 | value 各列尺度 2^-8 到 2^8，gate 各维含 -16/-8/0/8 |
| retention | 256、8192、65536 | q/k 为单位基向量、v=0、g=-16、beta=-12，初始活动状态为 1 |
| varlen | 总计 1456 | 独立长度 15/16/17/127/128/129/1024，覆盖 chunk 边界和尾部 |

## 四种数值口径

1. **FP64 reference**：从同一组 BF16 raw 输入开始，在 FP64 中做带 1e-6 的 q/k 归一化、
   精确 sigmoid/gate 和逐 token 递推。初始状态先舍入到 BF16，与官方内部可表示值对齐。
2. **official**：官方 BF16 内部状态、MMA、激活和中间转换的总误差，相对上述 reference。
3. **bf16_checkpoint / fp32_checkpoint**：所有运算仍为 FP64，只在每 16 token 后以及序列末尾
   将状态舍入到指定格式再转回 FP64。输出不额外量化，用来隔离状态存储误差。
   这两者不是新实现的完整 CUDA 内核，也不是官方 FP32 API 的两个版本。
4. **output_rounding_floor**：reference 输出只在最后转一次 BF16，展示输出表示误差的量级；
   这是参考基线，不是对不同算法的严格逐元素误差下界。

## 验证与判据

- FP64 token 递推对提供的 `fla_kda_ref/naive.py` 做 3 seeds × 5 个边界长度的校验。
  原参考内部为 FP32，要求输出/状态相对 L2 差均小于 2e-5。
- retention 另有闭式解，要求 FP64 最终状态绝对差小于 1e-10。
- 全部 case 对比 FP32 与 BF16 状态 API：FP32 初始值保留未量化精度，BF16 API 使用其舍入值。
  要求输出逐位相等、FP32 最终状态等于 BF16 最终状态提升到 FP32，验证接口转换语义。
- T=8192 的所有固定长度 case，以及 retention T=65536，检查按 256/1024 token 分段续算，
  要求与整段输出/状态逐位相同。另测首段 17 token，记录重排 chunk 边界造成的差异，不要求逐位相同。
- varlen 每条序列与单独调用比较，要求输出/状态逐位相同。
- 每个 seed 分别记录整体、每 256 token 输出窗口，以及若干前缀最终状态的误差。
  指标为 relative L2、RMSE、reference RMS、max/p99 absolute error、最差 value 列 relative L2 和 finite。
  relative L2 的分母最低取 1e-30，近零参考必须同时读绝对误差。
- 运行前固定 **2% relative L2** 为输出/状态筛查阈值，保留 `within_2pct`，不筛掉失败样本。
  该阈值不是模型质量标准；stress 失败是需要分析的结果，流程完成不等于所有 case 通过。

## 文件

| 文件 | 内容 |
|---|---|
| `run.py` | 输入矩阵、官方对照、续算检查、误差统计与复现入口 |
| `reference.py` | 可读的 FP64 token 递推及仅状态舍入模型，用 TorchScript 降低 CPU 循环开销 |
| `probe.py` | 单 chunk、零状态、beta 扫描；读出官方 K1 inverse，对照 FP64 三角求解 |
| `results/outputs.csv` | 各 seed/sequence 整体输出误差 |
| `results/windows.csv` | 每 256 token 输出窗口，避免长序列平均掩盖尾部误差 |
| `results/states.csv` | 16/64/256/1024/4096/8192/16384/32768/65536 等前缀的状态误差 |
| `results/checks.csv` | API、流式和 varlen 的对照 |
| `results/reference_checks.csv`、`retention.csv` | 参考校验、闭式解及反例状态值 |
| `results/metadata.json` | 环境、判据、代码/扩展哈希及完成状态 |
| `results/probe/` | 18 组单 chunk 对照及独立采集元数据 |

普通 case 的状态布局在官方接口中为 `[V,K]`，数学参考中为 `[K,V]`，比较前显式转置。
不同 T 的随机数据不完全共享前缀；误差随位置的增长分析使用同一长序列的 `states.csv` 和窗口记录。
