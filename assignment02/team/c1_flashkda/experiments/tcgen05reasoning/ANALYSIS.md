# 讨论点 2：CHUNK=16 能否直接受益于 tcgen05

**结论：CHUNK=16 不排斥 tcgen05，但当前简单替换没有普遍收益。** 小方阵有 75% 的算术 padding；
长方形转置后可无 padding。小 batch 单次调用均未获益，大 batch 的转置投影达到 1.082×，
驻留累加的状态更新达到 1.140×。这些是局部收益，尚不是完整 FlashKDA 加速。

## 1. 纸面推算：小 CHUNK 不等于所有矩阵乘都小

讨论限定 BF16 输入、FP32 累加、dense `tcgen05.mma.cta_group::1.kind::f16`。
该路线的 M 可为 64/128，N 为 8 的倍数，K 的指令步长为 16。
这里的 `kind::f16` 同时覆盖 FP16 和 BF16，不能把名称理解为只支持 FP16。
官方 SM80 atom 是 `m16n8k16`。
依据：[NVIDIA tcgen05 MMA API](https://docs.nvidia.com/cutlass/4.2.1/media/docs/pythonDSL/cute_nvgpu_tcgen05.html)、
[PTX MMA 说明](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-mma-instructions-mma)，
以及本地 [SM80 调用](../../FlashKDA/csrc/smxx/fwd_kernel2.cuh)。

令 C=16、D=128。以下利用率仅计算有效 M×N×K / 实际 M×N×K，
不是测得的 Tensor Core 利用率，也不包含三角掩码导致的数学零。

| 运算 | 原始 M,N,K | tcgen05 原方向 | 有效算术占比 | 转置 Cᵀ=BᵀAᵀ 后 | 有效算术占比 |
|---|---|---|---:|---|---:|
| Gram / Mqk | 16,16,128 | 64,16,128 | 25% | 64,16,128 | 25% |
| K/Q × state | 16,128,128 | 64,128,128 | 25% | 128,16,128 | 100% |
| INV/Mqk × U | 16,128,16 | 64,128,16 | 25% | 128,16,16 | 100% |
| 状态更新 | 128,128,16 | 128,128,16 | 100% | 128,128,16 | 100% |
| 方阵乘形状探针 | 16,16,16 | 64,16,16 | 25% | 64,16,16 | 25% |

因此“CHUNK=16，所以 tcgen05 全部只有 25% 利用率”不成立。
两类长方形运算可以交换操作数并转置输出，以 feature 轴作为 M，保持 CHUNK 不变。
这改变布局与 epilogue，不改变矩阵乘的数学含义；没有把多个 head/chunk 拼成一个大 tile。
16×16 方阵在这种转置下仍然过小。

指令数量也不能直接等价为耗时：

| 运算 | SM80 m16n8k16 atom / 矩阵 | tcgen05 MMA / 矩阵（两种方向） |
|---|---:|---:|
| gram | 16 | 8 |
| projection | 128 | 8 |
| local_mix | 16 | 1 |
| state_update | 128 | 1 |
| square | 2 | 1 |

公式分别为 `(M/16)(N/8)(K/16)` 与 `K/16`，每次仅计算一个完整结果。
SM80 测试通过 WMMA 16×16×16 生成 HMMA；表中按底层 16×8 atom 计数。
这些是无展开重复的逻辑工作量，`repeats=32` 时再乘 32。

## 2. 换指令必然同时改变数据通路

SM80 从寄存器 fragment 读取并把累加结果留在寄存器；本 tcgen05 路线从 shared memory
读取 A/B，把累加结果放入 TMEM，完成后再显式加载出来。后者至少引入 TMEM
分配/释放、异步完成通知及等待、TMEM → 寄存器回读。
参考：[NVIDIA tcgen05 编程指南](https://docs.nvidia.com/cutlass/4.5.2/media/docs/pythonDSL/mma_docs/tcgen05_programming.html)。
所以这里的“只换指令”是保持 CHUNK 和数学运算不变，改写对应数据通路；不是只替换一行汇编。

本实现原方向将 M<64 补零；转置方向能避免长方形补零。
所有输入都合并读取全局内存。SM80 shared leading dimension 加 8 以缓解 bank conflict；
tcgen05 用 K-major 32B/128B swizzle。原方向 tcgen05 用 warp 内 shared tile 重排回写，
转置方向直接合并写回；这些成本均在计时内。

TMEM 分配也不能等同于有效输出大小：本实现 N=16 时申请 32 列、N=128 时申请 128 列。
因此一个 16×16 FP32 输出只有 1 KiB，却申请 128×32×4=16 KiB 的 TMEM 地址容量。
这不等于每个输出都产生 16 KiB 的访存，也不能只靠它推导运行时间。
M=64 的输出位于每组 32 条 TMEM datapath 的前 16 条，读取映射参见
[CUTLASS tmem_frg](../../FlashKDA/cutlass/include/cute/atom/mma_traits_sm100.hpp)。

## 3. 实验设计与判据

[复现说明](README.md) 给出完整参数。比较七个实现：SM80 原方向 1/4/8 warp、
SM80 补零/转置控制组，以及 tcgen05 原方向/转置。
所有实现使用相同的量化输入与 FP32 输出，五类形状逐项对 FP64 GEMM。
`small integer/zero` 必须精确相等，随机输入用整体相对 Frobenius 误差检查。

两种测量回答不同问题：

- `repeats=1`：独立局部算子的完整开销，包含 staging、分配、同步和输出。
- `repeats=32`：输入装载一次，同一个结果连续执行 D+=AB，最后回写一次。
  这检查固定开销摊薄后的方向，但不包含真实 KDA 的中间状态更新、激活、BF16 舍入与依赖。

batch=12/96 检查小规模独立矩阵，接近 TP8/TP1 的 head 数量。
batch=6144/49152 是 8192/16×12/96 个独立 chunk-head，适合观察高并行局部吞吐；
不能宣称真实 K2 同时拥有这么多独立任务。其状态跨 chunk 串行依赖仍然存在。

性能比较只使用同次运行。所有方法共享输入和输出缓冲区，预分配、热缓存、CUDA Graph
计时；每轮随机方法顺序，记录轮次分布与 GPU 频率。SM80 基线取已测原方向和转置候选的
最快值，避免用单一较差 warp 配置人为制造 tcgen05 加速。

## 4. 实测结果

正式运行：[配置](results/metadata.json) · [完整汇总](results/SUMMARY.md) · [原始计时](results/timing.csv)。
B300，148 SM，Slurm job 25093，GPU UUID `GPU-778768b4-6c9e-e483-890e-0812760948ae`。
223 次时钟采样均为 SM **1095 MHz**、显存 3996 MHz；采样间隔 200 ms，不能排除间隔内变化。
未调整频率或功耗。以下是这张卡在该频率下的同次比较，不与讨论点 1 的 2032 MHz 绝对耗时横比。

630 组数值检查通过，最大整体相对误差为 **4.537e-6**，整数和零输入精确匹配。
[memcheck](results/memcheck.log) 0 errors，[racecheck](results/racecheck.log) 0 errors / 0 warnings。
[静态反汇编](results/instructions.json) 确认生成 `HMMA.16816.F32.BF16` 与 `UTCHMMA`，
本次共有 1400 行计时统计，即 280 个配置各五轮。

### 小批量单次调用：没有测得正收益

下面取 batch=96、repeats=1；单位 µs/kernel。speedup=最佳 SM80 / 最佳 tcgen05，大于 1 才是加速。

| 运算 | 最佳 SM80 | tc 原方向 | tc 转置 | speedup | 同轮比值范围 |
|---|---:|---:|---:|---:|---:|
| gram | 6.624 | 13.009 | 12.532 | 0.529× | 0.527–0.532× |
| projection | 9.738 | 20.174 | 15.948 | 0.611× | 0.610–0.612× |
| local_mix | 4.473 | 10.486 | 7.541 | 0.593× | 0.588–0.596× |
| state_update | 9.458 | 15.119 | 12.938 | 0.731× | 0.728–0.732× |
| square | 2.018 | 4.422 | 4.138 | 0.488× | 0.485–0.488× |

五项的最佳 SM80 均为 8 warp；batch=1/12 也都没有测得 tcgen05 单次调用收益。
转置显著改善 projection/local_mix，但避免补零并不保证覆盖数据通路成本。
对小方阵，既有形状浪费，又有一次调用必须支付的 TMEM/同步/回读开销。
这里只测了整体 kernel 时间，没有把各项开销逐条拆分，因此不把差值全部归因于某一条指令。

### 大批量：projection 出现局部反例

固定 batch=49152、repeats=1，单位 µs/kernel：

| 运算 | 最佳 SM80 | SM80 µs | 最佳 tcgen05 µs | speedup |
|---|---|---:|---:|---:|
| gram | sm80_w1 | 359.452 | 765.999 | 0.469× |
| projection | sm80_w8 | 1939.090 | 1792.581 | 1.082× |
| local_mix | sm80_w1 | 243.748 | 323.708 | 0.753× |
| state_update | sm80_w8 | 1116.870 | 1720.452 | 0.649× |
| square | sm80_w8 | 59.278 | 281.996 | 0.210× |

projection 在 batch=6144/49152 时分别达到 **1.057× / 1.082×**；原方向则仍慢。
这足以反驳“保持 CHUNK=16 的 tcgen05 局部算子不可能获益”，但 projection 属于 K2，
真实算法并不能把所有 chunk 的状态投影同时展开为本实验的大 batch。这个收益不能直接带回完整 KDA。

### 驻留累加：state_update 的方向发生变化

取 batch=96、repeats=32；仍以一次 kernel 的总时间比较，输入只装载一次、最后回写一次。

| 运算 | 最佳 SM80 µs | 最佳 tcgen05 µs | speedup |
|---|---:|---:|---:|
| gram | 18.812 | 22.909 | 0.821× |
| projection | 22.637 | 26.461 | 0.855× |
| local_mix | 5.449 | 8.921 | 0.611× |
| state_update | 16.975 | 14.887 | 1.140× |
| square | 2.704 | 5.507 | 0.491× |

state_update 从单次调用的 0.731× 变为 **1.140×**，五轮同轮比值为 1.137–1.142×。
batch=1/12 的驻留累加也有约 1.16× 收益。它说明应考察长驻累加、融合和复用，
不能仅凭独立 kernel 调用就否定 tcgen05。这里连续累加同一 A/B，没有逐 chunk 生成新 U、
状态衰减和中间舍入，故不是已经完成 K2 的融合改造。

## 5. 控制组与资源证据

相同 4 warp 的 SM80 补零/转置控制，batch=96、repeats=1，单位 µs：

| 运算 | 原方向 | M 补到 64 | 转置 |
|---|---:|---:|---:|
| gram | 10.892 | 11.507 | 10.953 |
| projection | 15.961 | 21.493 | 16.151 |
| local_mix | 7.139 | 8.914 | 7.136 |
| state_update | 15.395 | 15.394 | 15.601 |
| square | 2.843 | 3.107 | 2.903 |

补零确实有代价，但不是“算术量四倍，所以耗时必为四倍”。例如 gram 增加约 5.6%，
projection 增加约 34.7%；这些变化还包含 staging、shared footprint 与 CTA 内工作分配。
当 M 已为 128 时，state_update 的原方向与补零控制编译为相同形状，实测也基本一致。

[资源记录](results/resources.csv) 显示全部 kernel 的 local memory 为 0，编译日志也未报告 spill。
CUDA occupancy API 对本次所有 tcgen05 配置都返回 **1 CTA/SM**；SM80 返回 3–32，
具体依寄存器、shared memory 和线程数而变。例如 square 的最佳 SM80 为 8 CTA/SM，
tcgen05 为 1；gram 大 batch 的最佳 SM80 为 20，tcgen05 为 1。
因此小任务大批量吞吐还受到调度与驻留差异影响，不能把结果仅解释为 tile padding。
这是当前 B300/CUDA 13 上的 API 结果，不是本实验对所有架构/驱动的保证；
也不是动态测得的 achieved occupancy，不能单凭它判定唯一瓶颈。

## 6. 对讨论点 2 的回答

1. **形状并非全面失配。** 16×16 方阵至少补到 M=64，有效算术占比 25%；
   16×128 的长方形可转置为 128×16，状态更新本来就是 128×128，均可无算术 padding。
2. **没有测得简单独立调用的普遍收益。** batch=1/12/96、一次乘法时五类运算均慢于最佳 SM80 候选，
   支持谨慎保留当前 SM80 路线；不能据此证明所有 tcgen05 实现都不可能更快。
3. **保持 CHUNK=16 仍有局部收益空间。** 大 batch 的转置 projection 达到 1.082×，
   驻留累加的 state_update 达到 1.140×（batch=96）。收益依赖布局、工作量和固定成本的摊薄。
4. **下一步应测真正融合的 K2 切面。** 固定 CHUNK=16，选状态更新或投影，保留真实的状态依赖、
   dtype/舍入、decay 和输出接口，再和官方 kernel 对拍、比较端到端时间。不能相加本页时间预测加速。

本实验不是 FlashKDA 算法级正确性验证；square 使用 BF16×BF16→FP32，
也不替代官方 FP16 累加、融合 Neumann 求逆的精度/性能测试。
未覆盖 TMA 多级流水、更多布局与 warp 数、跨 head 打包、2-CTA 或完整 persistent K2；
本页结论限于已列候选和测量范围，不宣称指令的性能上限或全局最优方案。

