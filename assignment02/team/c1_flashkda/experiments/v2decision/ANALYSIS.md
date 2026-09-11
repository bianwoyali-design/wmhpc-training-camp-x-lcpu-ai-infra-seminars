# 讨论点 6：v2 出不出 Blackwell 专版？

**我的选择：保留现有通用实现，开发可选、按形状选择的 Blackwell 优化后端；
不把 tcgen05 全量替换作为 v2 默认方案。**
任务 3 已完成一个真实的并行度重构切面：value-split K2 在 B300 的 H12 单长序列上取得
**1.208× 完整 forward 加速**，但 H96 和多序列组有明显回退。它支持形状分派，
不支持一个专版覆盖所有场景，也不证明收益必须依赖 tcgen05。

## 任务 3：实际实现了什么

[build.py](build.py) 对固定快照生成可检查的 CUDA 改动，保留官方 K1 和 K2 的逐 chunk 运算，
将一条 sequence/head 的 value 列分给 P=2/4 个独立 CTA。各分区保留完整 K 维，
只更新自己的状态行（外部布局 `[V,K]`），没有把时间依赖切断，也不需要跨 CTA 归约。

每个分区使用 2/1 个 compute warp，加一个 load warp 和一个 store warp；
输出与最终状态由 store warp 合并写回自己的分区，barrier 参与数同步修改。
仍使用 SM80 MMA 和 TMA，编译目标是本机 `sm_103a`。这是任务允许的**并行度重构**，
不是 tcgen05 内核，也不是只跑局部 GEMM 的 microbench。

为控制实现范围，输入 TMA tile 和 shared 缓冲没有缩小：各 CTA 仍加载全宽输入及初始状态，
动态 shared footprint 仍为 **98,432 B**。因此 P=4 是增加独立 CTA 并减少每 CTA 的有效计算，
同时承担重复加载成本，不是理想的零成本四路拆分。

构建四个库：p0 重编译官方；p1 只改写回方式；p2/p4 再进行 value 拆分。
它们和已安装 official 扩展一起跑完整 forward，包括 beta 转置、tile prefix、K1 和 K2。
支持范围及[复现命令](README.md)明确限定 D128、BF16 状态、varlen 接口；未做公开 API 或自动 dispatch。

## 正确性与正式性能

正式运行 Slurm **25528**，B300、148 SM，UUID `GPU-dadf9f3b-df58-d3fa-07b0-5fe223423db1`。
57 次频率采样中 55 次为 2032 MHz，另外两次为 2010/2025 MHz。未锁频，采用同次运行随机方法顺序。

- **50 项** case/方法检查的输出与最终状态均和 official 逐位一致且有限，含三组状态压力输入。
- 短边界组另有 **20 项** FP64 输出/状态对拍，最大 relative L2 为 **0.6955%**；
  FP64 参考先通过提供的 `fla_kda_ref/naive.py` 校验。
- memcheck **0 errors**，racecheck **0 errors / 0 warnings**。
  范围为五种 K2 各一次边界组调用，不是全部形状穷举。
- 四个重编译 K2 都无 spill。p0/p1/p2/p4 分别为 73/75/78/90 registers/thread，
  192/192/128/96 threads/CTA。更多分区不自动降低每线程寄存器数。

五轮中位数的中位数，单位 µs/full forward；不是 KDA 整层或模型端到端时间：

| workload | official | p0 | p1 | p2 | p4 | 最佳拆分相对 official |
|---|---:|---:|---:|---:|---:|---:|
| H3，长度 15/16/17/33 | 20.685 | 20.662 | 27.670 | 19.469 | 17.251 | 1.199× |
| H12，1×8192 | 781.846 | 781.859 | 1879.066 | 834.506 | 647.069 | **1.208×** |
| H12，8×1024 | 144.246 | 144.147 | 287.024 | 179.366 | 209.958 | 0.804× |
| H96，1×8192 | 1018.122 | 1018.134 | 2110.326 | 1266.128 | 1489.155 | 0.804× |
| H96，8×1024 | 682.883 | 682.579 | 1078.390 | 1003.085 | 1241.040 | 0.681× |
| H12，1×128 | 24.333 | 24.368 | 48.483 | 28.883 | 23.174 | 1.050× |
| H96，4608 + 7×512 | 1000.144 | 1000.595 | 1831.651 | 1303.725 | 1479.674 | 0.767× |

H12 单长序列 p4 的五轮配对加速比为 **1.2080–1.2085×**，不是挑一轮的最好值。
H96 N8 的 p4 只有 **0.5498–0.5548×**，这类回退也很稳定。
原始记录见 [timing.csv](results/timing.csv)、[checks.csv](results/checks.csv)、
[reference.csv](results/reference.csv)、[resources.json](results/resources.json)。

### 收益与回退怎么解释

p0 与 official 基本重合，说明收益不是选了一个明显更差的官方编译版本作为基线。
p1 在 H12 N1 从 782 增至 1879 µs，表明改变写回/同步的数据通路本身代价很大。
p4 在相同写回机制下缩小每 CTA 的计算和写回范围，最终仍优于原始 TMA 基线；
不能仅用 p1/p4≈2.90× 宣称“官方被加速 2.90 倍”。

H12 N1 从 12 个 CTA 增至 48 个 CTA，符合[讨论点 3](../parallelismreasoning/ANALYSIS.md)
提出的低并行度改善方向。H96、多序列已经有更多任务时，重复加载、保留的完整 shared 缓冲、
额外 CTA 和写回成本反而不划算。这里有控制组与资源证据，但没有对新内核逐项做 NCU 因果分解，
不能把回退全部量化归因于 HBM 或某一种 barrier。

任务 3 的结论是：**当前完整实现取得了局部正收益，也明确找到了回退域。**
不是“所有 value-split 实现的最优速度”，也不是“SM80 指令已经到性能上限”。

## 前五个讨论点如何约束 v2

| 证据 | 对 v2 的含义 |
|---|---|
| [1：范围与 CHUNK](../chunk16reasoning/ANALYSIS.md)：未 rescale 的 C32/64 最坏指数 ±160/±320 越界 | 不能为配大 tile 直接增大 CHUNK；更换成 FP32 也不解决指数范围 |
| [2：tcgen05](../tcgen05reasoning/ANALYSIS.md)：16×16 方阵只有 25% 有效算术，但长方形转置可无 padding | 不应全面否定新指令，也不应把所有阶段统一替换 |
| 2 的小 batch 独立调用均无收益，大 batch projection 1.082×、驻留 state_update 1.140× | 支持进一步做真实融合切面；这些不是完整 K2 的加速比 |
| [3：并行度](../parallelismreasoning/ANALYSIS.md)：K2 的独立状态链不足，多 head/CTA 和软件队列不能凭空造任务 | 先按每卡 NH 和长度分布选择组织；本次挑战验证了部分 value-split 假设 |
| [4：瓶颈](../bottleneckreasoning/ANALYSIS.md)：低 NH 时空闲 SM 多；高 NH 时 shared/issue 压力明显 | 峰值 Tensor FLOPS 不是唯一目标；TMEM/布局/数据供给成本必须一起算 |
| [5：精度](../stateprecisionreasoning/ANALYSIS.md)：小更新可使状态停在 1，FP32 API 仍用 BF16 内部状态；相关输入有 inverse 异常 | 精度修正必须独立验收，不能用“无 NaN”或 FP32 接口 dtype 代替验证 |

本次 value-split 与 official 逐位相同，也会继承讨论点 5 的精度问题。
“与基线一致”只满足重构的等价性要求，不意味着所有输入域达到模型精度要求。

## 为什么值得做 Blackwell 优化后端

第一，真实低 NH 负载有可测的优化空间，而不仅是规格表上的峰值差。
第二，CHUNK=16 的 projection/local_mix 可换方向匹配较大的 M，state update 也不需要算术 padding，
tcgen05 仍值得在真实状态依赖下融合试验。
第三，v2 可以根据设备、NH、序列长度/不均衡程度选择实现，把局部收益留给适合的负载。

但是，当前获益的 value 拆分在数学上并不只适用于 Blackwell；是否也适合 Hopper 需要实测。
应尽量共享算法和接口，按设备特征替换内核，而不是为“专版”复制整个代码库。

## 为什么不默认全面换成 sm100a/tcgen05

**性能证据不够。** 本次挑战在 H96 和多序列上退化；tcgen05 实验也只有局部受益。
不能将独立 GEMM 时间相加预测 KDA，更不能假定峰值翻倍就能让依赖链缩短一半。

**可移植性有真实成本。** 当前官方虽用 SM80 MMA，仍使用 TMA 等较新功能，
不能将“SM80 atom”理解成整个库可以直接在 A100 上运行。
本地 setup 声明 90a/100a/103a/120a 构建目标，这个列表不等于本项目已经在全部设备验证。

架构专用 `a` 目标不具有一般的前向/后向兼容保证；本机 B300 用 `sm_103a`，
不能只发一个 `sm_100a` cubin 就假定覆盖所有 Blackwell。
family-specific `f` 目标提供同族兼容的另一条路线，但仍需核对实际用到的指令及工具链。
依据：NVIDIA [Blackwell compatibility guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/)、
[family-specific targets](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features)。

因此维护成本至少包括：各目标编译与装载检查、设备分派与回退、不同资源上限、布局/TMEM/异步同步回归，
以及尾部、varlen、状态 dtype 和精度域的交叉验证。本实验只在 B300 实测，不能把 H100/GB200/RTX 的表现写成既定事实。

## 发布决策与进入默认路径的条件

| 路线 | 当前决定 | 进入发布的条件 |
|---|---|---|
| 现有内核 | 保留为性能回退路径 | 保留其已知精度边界，不能作为所有压力域的精度兜底 |
| 当前 p4 value-split | 保留为实验后端；优先复验 H12 N1 | 更多独立输入、目标设备与服务形状复测，验证实际 dispatch 后仍有 full-forward 收益 |
| tcgen05 融合 K2 | 继续研发，暂不默认 | 保留真实 decay/状态/输出，完成算法对拍、sanitizer 和同次端到端对照 |
| 大 CHUNK | 暂缓 | 完整 rescale/求逆/状态精度设计完成后再测，不能只有范围 microbench |
| 精度保护路径 | v2 必须明确设计 | 真正更高精度状态或补偿、稳定求逆及真实模型激活/质量验证；目前尚未实现 |

建议性能发布门槛为预先指定的目标形状 full-forward 至少快 10%，多轮和多个输入种子复验；
不适用形状明确回退，禁止用事后挑选形状后的平均数掩盖退化。这个 10% 是发布策略建议，
不是现有实验的统计置信界或既有要求。
精度门槛需根据模型用途确定：讨论点 5 的 2% 只是合成筛查，不能代替任务质量标准。

**最终回答：出可选、按形状分派的优化后端，暂不出默认全量替换的 tcgen05 专版。**
官方保留 SM80 MMA 有充分工程理由；本次 1.208× 的完整挑战又说明当前实现仍有可改进空间。
这两个结论可以同时成立。
