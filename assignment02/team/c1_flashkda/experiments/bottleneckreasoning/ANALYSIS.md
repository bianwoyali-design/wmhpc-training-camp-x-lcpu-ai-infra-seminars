# 讨论点 4：不要把 FlashKDA 当作一个大 GEMM

本实验分析官方 CHUNK=16、D=128、BF16 state 的 K1/K2，覆盖每卡 H=12/96、
单序列、八条等长序列和八条不等长序列。复现入口与测量口径见 [README](README.md)。

**结论：不能统一归为 Tensor compute-bound 或 HBM bandwidth-bound。**
K1 是指令发射与访存/同步混合限制；K2 在低并行度时主要受独立 CTA 数和递推延迟限制，
并行度增加后出现更明显的片上访存压力。这里的“访存”必须区分 HBM 与 L1TEX/shared memory。
没有哪组证据支持“只换成更高峰值 Tensor 指令，就能按峰值比例加速”。

## 正式测量

主实验为 Slurm **25203**，B300 SXM6 AC、148 SM，GPU UUID
`GPU-ab0e8f8b-abe0-a03f-c8e9-65622211a5f4`。PyTorch 2.13.0+cu130，NCU 2025.3.1。
`nvidia-smi` 的 1390 次采样中 1389 次为 2032 MHz；NCU 的 GPC 平均频率为约
1893–1985 MHz。两种观测口径分别保留，没有锁频，也没有把低频 pilot 混入正式结果。

未插桩的热缓存 Graph 计时如下，各取五轮中位数的中位数，单位 µs：

| H | 序列长度 | full | K1 | K2 |
|---:|---|---:|---:|---:|
| 12 | 8192 | 788.74 | 42.15 | 742.99 |
| 12 | 8×1024 | 144.48 | 42.11 | 101.47 |
| 96 | 8192 | 1027.29 | 273.60 | 750.82 |
| 96 | 8×1024 | 677.16 | 273.51 | 402.47 |
| 96 | 4608 + 7×512 | 1005.15 | 282.30 | 720.05 |

full 与隔离阶段的缓存状态和调度开销不同，不能要求三列严格相加。
原始轮次见 [normal_timing.csv](results/normal_timing.csv)。

以下采用 cold NCU 的统一口径；百分比按对应硬件单位峰值归一化，Tensor 为 elapsed：

| case | 阶段 | SM % | Tensor % | DRAM % | L2 % | L1TEX % | eligible warp/SMSP |
|---|---|---:|---:|---:|---:|---:|---:|
| H12 N1 | K1 | 57.62 | 4.61 | 33.63 | 58.20 | 55.20 | 2.09 |
| H12 N1 | K2 | 2.49 | 2.34 | 2.13 | 3.34 | 4.63 | 0.34 |
| H12 N8 | K2 | 19.18 | 17.97 | 16.13 | 25.86 | 35.91 | 0.33 |
| H96 N1 | K1 | 70.15 | 5.62 | 58.69 | 72.11 | 67.57 | 2.26 |
| H96 N1 | K2 | 20.54 | 19.34 | 18.68 | 27.11 | 39.02 | 0.34 |
| H96 N8 | K2 | 38.90 | 36.31 | 36.28 | 53.23 | 69.84 | 0.58 |
| H96 skew8 | K2 | 21.43 | 20.12 | 20.25 | 29.62 | 38.29 | 0.52 |

完整二十行 cold/warm 对照见 [SUMMARY.md](results/SUMMARY.md)，实际流量、时钟、
stall、active/elapsed 对照见 [summary.csv](results/summary.csv)。

### K1：有明显访存压力，但不是 Tensor 算力吃满

H96 N1 的 SM 70.15% 与 issue elapsed **70.15%** 一致，Tensor 只有 **5.62%**；
ALU/LSU issue 各约 40%，不能把顶层 SM % 解读为矩阵乘算力利用率。
L2 72.11%、L1TEX 67.57%、DRAM 58.69% 同时较高，实际 DRAM 流量约 **4.50 TB/s**。
eligible 为 2.26，achieved occupancy 为 96.61%，所以这里不像 K2 那样主要缺少独立 CTA。

barrier 38.44%、long scoreboard 19.00% 提示阶段同步和数据供给都值得检查，
但这些是 warp 状态比例，不能据此承诺消除 38% 的运行时间。
合理分类是 **issue 与内存系统/同步的混合限制**；更值得试减少中间流量、标量指令和同步，
而不是只提高 MMA 峰值。仅凭约 59% 的 DRAM 值还不能证明 HBM 是唯一瓶颈。

### K2：先看有多少独立状态链，再看活跃 SM 在等什么

K2 每条 sequence/head 一个 CTA，192 threads，动态 shared memory 98,432 B；
shared memory 允许最多 2 CTA/SM，理论 occupancy 上限仅 12/64=18.75%。

| case | CTA 数 | SM active/elapsed % | Tensor active % | achieved occupancy % |
|---|---:|---:|---:|---:|
| H12 N1 | 12 | 7.68 | 30.52 | 9.37 |
| H12 N8 | 96 | 61.90 | 29.04 | 9.33 |
| H96 N1 | 96 | 63.81 | 30.31 | 9.37 |
| H96 N8 | 768 | 90.45 | 40.14 | 16.82 |
| H96 skew8 | 768 | 53.46 | 37.63 | 15.02 |

- **H12 N1**：只有 12 条状态链，148 个 SM 大多没有工作。DRAM 只有 2.13%，
  不能称为 HBM 带宽饱和。H12 N8 在相同 token/head 总工作量下快 **7.32×**，
  主要变化是独立状态链增多、每条变短；K1 时间基本不变。
- **H96 N1**：比 H12 N1 多做 8 倍 Tensor 工作，K2 却仍约 751 µs，
  支持空闲硬件被更多 head 填充的解释。其 96 个 CTA 仍少于 148 个 SM。
- **H96 N8**：768 个 CTA 让 SM active 升至 90.45%，但只有约 0.58 个 eligible warp/SMSP，
  wait 18.79%、short scoreboard 15.86%、long scoreboard 10.94%。
  递推、片上数据依赖和 warp 角色配合仍限制延迟隐藏；Tensor 与 DRAM 都没有接近各自峰值。
  L1TEX 接近 70%，需要细分计数器，不能继续仅用“CTA 太少”概括。
- **H96 skew8**：CTA 数和总 Tensor 工作量与等长 N8 相同，K2 却慢 **1.79×**，
  SM active 从 90.45% 降至 53.46%。这支持长短序列负载不均衡/尾部的解释；
  两者也有 varlen 路径差别，[讨论点 3](../parallelismreasoning/ANALYSIS.md)另有等长 varlen 控制组。

增加独立序列是工作负载对照，不能将已有的一条状态链任意断开。
下一步内核优化应验证 value 维拆分、shared 数据供给和状态依赖的取舍，而不是强行拆时间。

### 缓存对照与证据边界

细分补采见 [memory/metrics.csv](results/memory/metrics.csv)：Slurm **25209**，
GPU `GPU-3924bd78-8b2f-8e85-3f20-8b0687d0bb0a`，NCU GPC 约 1076–1090 MHz。
这是另一张较低频的 B300；以下比例只在补采内部解释，不把其耗时与主实验直接比较。

| 补采 case | 阶段 | LSU wavefront 吞吐 % | shared wavefront / 全部 LSU wavefront |
|---|---|---:|---:|
| H96 N1 | K1 | 71.32 | 96.28% |
| H96 N8 | K1 | 71.45 | 96.31% |
| H96 N1 | K2 | 38.02 | 99.62% |
| H96 N8 | K2 | 69.63 | 99.51% |

使用 `l1tex__data_pipe_lsu_wavefronts_mem_shared.sum / l1tex__data_pipe_lsu_wavefronts.sum`
计算最后一列。L1TEX 的高值主要来自 shared 数据路径，不能解释成 HBM 已满。
H96 N8 K2 的 shared load/store bank-conflict 计数约为 279 万/563 万，说明存在冲突；
尚未通过 layout 改动做消除对照，不能把这些计数换算成可获得的加速比。
L2 breakdown 中 K1 较高的分量是 tag requests，而非单看传输字节。

warm 对照中，H96 N1 的 K2 Tensor/DRAM 为 19.29%/19.51%，H96 N8 为 36.18%/37.77%，
与 cold 分类一致；K2 warm/cold 耗时比落在 **0.999–1.016**。
因此当前结论不是“NCU 清缓存才造成 K2 很慢”。

warm K1 不一定比 cold 快：H12 N1 从 42.43 到 43.55 µs，同时观测的 DRAM bytes 增加。
`none` 表示不主动清缓存，不表示零 miss；预热调用、输出快照和尚未回写的数据都影响缓存状态。
两种 replay 模式也不同，这个对照只检验分类稳健性，不是纯粹改变缓存容量的因果实验。
缓存及 replay 设置见 NVIDIA [NCU CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)。

## 先算工作量，再看哪一层资源实际忙

每个 chunk/head 中，FMA 按两次运算计：

| 阶段 | Tensor 工作 | 执行的 Tensor operations |
|---|---|---:|
| K1 | 两次 16×16×128 Gram，六次 16×16×16 Neumann MMA | 180,224 |
| K2 | 两次投影、一次状态更新，各 2×16×128²；两次局部混合，各 2×16²×128 | 1,703,936 |

这统计实际 dense MMA 的工作量，包括三角矩阵中执行的无效元素；不计标量激活、
指数、转换及加法，不是完整算法 FLOPs。源码入口：
[K1](../../FlashKDA/csrc/smxx/fwd_kernel1.cuh)、
[K2](../../FlashKDA/csrc/smxx/fwd_kernel2.cuh)。

[workspace 定义](../../FlashKDA/csrc/smxx/utils.cuh)给出每个 chunk/head 的中间结果为
`3×16×128×2 + 128×4 + 2×16×16×2 = 13,824 B`。
按源码载荷估算：

- K1 读取 q/k/g 共 12,288 B，beta 的 TMA tile 为 64 B、dt_bias 512 B、A_log 4 B，
  写 workspace 13,824 B，共约 26,692 B；Tensor operations/载荷约 **6.75**。
- K2 读取 workspace、v、beta，写 out，共 `13824+4096+64+4096=22,080 B/chunk/head`；
  再加每条 sequence/head 初始和最终状态合计 **65,536 B**。
  不计状态时约 **77.17 operations/B**，计入状态后略低。

这是源码层面的载荷估计，未计描述符、地址元数据、对齐及事务放大；共享的 bias、beta
重叠读取可能命中缓存。workspace 写入也可能在当前 kernel 结束后才回写 HBM。
因此不能拿这些字节数冒充 NCU 测得的 DRAM bytes。

两者的载荷比值都低于 handout 使用的 281.25 FLOP/B 平衡点，但这个比较只说明
“假定这些字节经过 HBM，带宽 roof 比 Tensor 峰值低”；并不说明执行已接近该 roof。
K2 的跨 chunk 状态驻留 shared memory，一条 sequence/head 的 chunk 串行推进，
这与大 GEMM 拥有大量独立输出 tile 的执行方式不同。

## 用哪些 NCU metric 回答

以下名称均在本机 NCU 2025.3.1 / B300 上实际采集，原值和单位见
[metrics.csv](results/metrics.csv)。

| 问题 | metric | 读法 |
|---|---|---|
| HBM 是否接近饱和 | `dram__throughput.avg.pct_of_peak_sustained_elapsed` | 与实际读写字节、时间一起看 |
| 实际 HBM 流量 | `dram__bytes_read.sum`、`dram__bytes_write.sum`、`gpu__time_duration.sum` | 二者字节和/秒得到 GB/s |
| Tensor 管线是否忙 | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 全 kernel elapsed 口径；不是营销峰值 FLOPS 达成率 |
| 工作是否铺满 GPU | `sm__cycles_active.avg.pct_of_peak_sustained_elapsed`、`launch__grid_size` | 与 CTA 数、SM 数及尾部不均衡对照 |
| 活跃 SM 内的 Tensor 使用 | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 与 elapsed 对照，区分空闲 SM 和活跃 SM 内等待 |
| scheduler 能否找到可发射 warp | `smsp__warps_eligible.avg.per_cycle_active`、`smsp__issue_active.avg.pct_of_peak_sustained_active` | eligible 低说明等待难以隐藏；不等于只缺更多 occupancy |
| SM 高百分比来自什么 | `sm__throughput.avg.pct_of_peak_sustained_elapsed`、`sm__issue_active.avg.pct_of_peak_sustained_elapsed`、各 `sm__inst_executed_pipe_*` | 拆开 issue、ALU、LSU、XU、FMA 与 Tensor，不能只读顶层 SM % |
| 片上访存压力 | `lts__throughput.avg.pct_of_peak_sustained_elapsed`、`l1tex__throughput.avg.pct_of_peak_sustained_elapsed`、`lts__t_sector_hit_rate.pct` | L2/L1TEX 忙不等于 HBM 忙；L1TEX 也服务 shared memory |
| warp 在等什么 | `smsp__warp_issue_stalled_{short_scoreboard,long_scoreboard,barrier,wait,mio_throttle,math_pipe_throttle}_per_warp_active.pct` | 用来提出下一步排查方向，不直接当作可消除的时间比例 |
| 是否受频率混淆 | `gpc__cycles_elapsed.avg.per_second` | 配合 `clocks.csv`；不同频率不可直接归因于代码 |

另采 achieved occupancy、shared bank conflicts、Tensor operation counts、寄存器和共享内存。
计数器定义参考 NVIDIA [Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)：
active 与 elapsed 分母不同；long scoreboard 涉及 L1TEX 依赖，short scoreboard 涉及
MIO 等依赖，都不能单独证明 HBM 带宽饱和。barrier 高也不代表删掉同步便能等比例加速。

## 与 assignment 4.5 的输入投影对照

`in_proj_qkvgfab` 是 **KDA 前面的输入投影**，不在本次 FlashKDA forward 中。
handout 的本地形状是 N=6288、K=7168，M 表示一次投影的 token 数量。
从 [4.5 表格](../../../../handout/src/assignment02.md)取代表行：

| M | AI，FLOP/B | %TCpeak | %BW |
|---:|---:|---:|---:|
| 1 | 1.0 | 0.2 | 66.1 |
| 16 | 15.9 | 3.8 | 67.8 |
| 256 | 237.8 | 45.1 | 53.4 |
| 4096 | 1842.7 | 76.3 | 11.6 |
| 16384 | 2781.0 | 78.3 | 7.9 |

小 M 时要搬约 90 MB 权重，却只有很少的 token 复用它；大 M 时权重复用增加，
更接近 Tensor 吞吐平台。这是用 AI 辅助判断的正例。
但[计时源码](../../../../cuda/m4_gemm/05_thin_gemm.cu)中的 GB/s 是
`2(MK+NK+MN)/time`，%BW 使用输入的 8000 GB/s，%TCpeak 使用 2250 TFLOPS。
它们是有效流量/名义峰值的模型，不是 NCU 的物理事务或 Tensor active cycles；
不要与本实验百分比直接相减，也不要把投影层的小 M 结论搬到 K2 上。

## 结论的适用范围

本实验只针对这五组 prefill 形状与 BF16 state，不覆盖 decode、FP32 state、所有 varlen
分布或整层 KDA 的输入/输出投影。每组每个缓存模式采一次 NCU，多个 replay pass 收集
不同计数器；正常计时有五轮。指标是瓶颈证据，不是严密的干预因果分解。
尤其不能仅凭某个 stall 最大，就宣布它是唯一瓶颈。

正式证据包含 12 个 NCU 报告（主实验 10 个、细分补采 2 个）、24 条 kernel 记录和
75 条正常计时轮次。所有 target 的输出/最终状态均有限，且与预热逐位一致。
主实验 Tensor operation 计数与上面的逐 chunk 成本一致；varlen K1 的 launch 含额外
越界保护 CTA，但有效 Tensor 工作量不变。两份 metadata 均为 complete，代码哈希匹配。
