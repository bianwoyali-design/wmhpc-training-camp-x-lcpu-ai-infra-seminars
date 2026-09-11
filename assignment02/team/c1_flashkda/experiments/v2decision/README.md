# 讨论点 6 与任务 3：完整 value-split K2 挑战

[决策与实测分析](ANALYSIS.md) · [端到端结果](results/SUMMARY.md)

本目录实际改变官方 K2 的跨 CTA 并行组织，保留 K1、CHUNK=16、SM80 MMA、BF16 状态和
逐 chunk 算术顺序。每条 sequence/head 的 value 列由 2 或 4 个普通 CTA 分担。
这是任务 3 的“并行度重构”路线，不是已经换成 tcgen05 的实现。

## 复现

需要本仓库的 FlashKDA/CUTLASS 快照、已安装官方扩展、CUDA 13、CUDA PyTorch、einops 和 B300。
从仓库根目录运行：

```bash
python3 assignment02/team/c1_flashkda/experiments/v2decision/build.py
srun -p gpu -G 1 --time=00:20:00 \
  assignment02/.venv/bin/python assignment02/team/c1_flashkda/experiments/v2decision/run.py
```

默认输出 `runs/<时间戳>/`；`--out` 可指定不存在的新目录，`--quick` 只测含尾部的短序列。
`build.py` 以断言检查补丁锚点，生成四个独立 shared library，均显式编译为 `sm_103a`。
生成的完整 CUDA 源码和编译日志位于 `build/p*/`；不修改官方源文件或已安装扩展。

## 实现与控制组

| 方法 | K2 组织 | 输出写回 | 用途 |
|---|---|---|---|
| official | 已安装官方扩展 | 官方 TMA/尾部路径 | 性能与逐位基线 |
| p0 | 重编译官方源码 | 同上 | 检查编译与 C ABI 包装差异 |
| p1 | 不拆分，4 个 compute warp | 一个完整 warp 合并写回 | 隔离写回改动的影响 |
| p2 | value 分两份，每 CTA 2 个 compute warp | 各写自己 64 列 | 并行度候选 |
| p4 | value 分四份，每 CTA 1 个 compute warp | 各写自己 32 列 | 并行度候选 |

load/store warp 各保留一个；producer/consumer barrier 的参与线程数随之调整。
每个 CTA 只计算、更新和写回自己的 value 列，K 维完整保留，无跨分区归约或状态交换。
输出覆盖每个 token 的对应 value 列，最终状态覆盖对应 `[V,K]` 行。

该版本为验证切面保留完整输入 TMA tile 和 shared 缓冲，包含初始状态的完整加载：
P 个分区会重复加载输入，动态 shared footprint 没有缩减。这里不声称实现了理想的缩小缓冲版本。
仅支持 D=128、BF16 非空初始/最终状态、varlen API；等长组也通过 cu_seqlens 调用，双方口径一致。
这些范围由测试入口保证，C ABI 是内部实验接口，不是带完整参数检查的公开 API。

## 测量

七组 workload：H=12/96、N=1/8、总 token=8192，H12 单序列 128 token，
H96 不等长 `[4608]+[512]*7`，以及 H3 边界长度 `[15,16,17,33]`。
另加 H3、T8192 的弱衰减、近恒等和小更新状态压力组，检查候选没有改变官方数值结果。
固定种子 42；所有候选共享输入，包含非零 BF16 初始状态。

- 所有 case/方法要求输出与最终状态对官方逐位相同且有限。初次输出预填 NaN，检测漏写。
- 短边界组另对讨论点 5 的 FP64 token 参考检查输出和状态，relative L2 均须小于 2%；
  该参考先对提供的 `fla_kda_ref/naive.py` 完成校验。
- 计时为完整 GPU forward：beta 转置、tile prefix、K1、K2，包含新的分区写回。
  预分配输出/workspace，每张 CUDA Graph 捕获五次 forward；五轮打乱方法顺序，各十次 CUDA Event 测量。
  不包含 Python dispatch 和输入分配，不代表完整 KDA 层或模型推理延迟。
- `p0` 与 official 比较控制编译/包装因素；`p1` 与 p2/p4 比较帮助解释分区与写回的交互。
  报告取五轮中位数的中位数，同时保存每轮 min/max。按 UUID 记录频率、功耗。

## 文件

`build.py` 是可复现的 CUDA 补丁与构建入口，`run.py` 是正确性和 full-forward benchmark。
`results/checks.csv`、`reference.csv`、`timing.csv`、`clocks.csv`、`metadata.json` 保存正式证据。
`build/` 和 `runs/` 为忽略的本地生成物。任务 3 的结果与是否默认发布专版的判断分开写在 ANALYSIS 中。

安全检查可分别运行下列命令（将 `memcheck` 换为 `racecheck` 检查共享内存竞争）：

```bash
srun -p gpu -G 1 --time=00:15:00 compute-sanitizer --tool memcheck \
  --error-exitcode 99 --kernel-name kns=_flash_kda_fwd_recurrence --launch-count 5 \
  assignment02/.venv/bin/python assignment02/team/c1_flashkda/experiments/v2decision/run.py \
  --quick --out assignment02/team/c1_flashkda/experiments/v2decision/runs/memcheck
```

过滤范围为边界组的五个方法各一次 K2，覆盖 official/p0/p1/p2/p4；不是全部形状的 sanitizer 穷举。
