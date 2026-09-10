---
title: 作业 2:Tensor Core & Pipeline
subtitle: Weiming HPC Training Camp $\times$ LCPU AI Infra Seminars · Session 3 / 4
---

# Preface {-}

这次作业主要是 Session 3 的配套练习，内容围绕 Tensor Core 展开，其中 Module 4 还会用到 Session 4 介绍的 tiling、TMA 和 pipeline。完成作业时主要参考课程课件和 PTX ISA 文档。

题型沿用系列惯例，并新增 DERIVE：先手工推导，再写程序验证推导。共七种：
CONCEPT 概念题、DERIVE 推导题、MODIFY 改造题、DEBUG 修 bug 题、
EXPERIMENT 实验题、HANDS-ON 动手题、FROM-SCRATCH 从零实现题。标
Optional 的为选做。涉及代码的题在标题行右侧标出文件路径(相对
`assignment02/`)，题面细节以文件头注释为准；FROM-SCRATCH 题。

硬件与构建:主线环境是 B300，`cuda/Makefile` 默认
`ARCH=100f`；M0、M1 也可以在 5090 上完成(`ARCH=120a make ...`)；
M2 的判测纯 host，无卡可判；M3、M4、M5 需要 B300。本作业统一用显式
`-gencode` 而不用 `-arch=sm_XXXa` 简写，原因见 `assignment02/README.md`，Makefile 已经配好。

实验纪律:所有性能数字都在自己占满的卡上测；测前测后
`nvidia-smi --query-compute-apps=pid，name --format=csv` 查有没有别的
进程占卡；可能 hang 的程序套 `timeout -k 5 <秒数>`(判测脚本已带)。

AI政策:必做题沿用仓库根目录 `CLAUDE.md`——AI 可以帮你理解、
review，不能替你实现；团队题不设限制。详见 `assignment02/README.md`。

## Content {-}

| Module | 主题 | 对应课件 | 代码位置 |
|---|---|---|---|
| 0 | 环境与峰值 | P1(S008--S021) | `cuda/m0_env/` |
| 1 | sm80:fragment 与 mma.sync | P2.2(S025--S041) | `cuda/m1_sm80/` |
| 2 | smem 供数:descriptor 与 swizzle | P2.3(S042--S070) | `cuda/m2_smem/` |
| 3 | sm100:tcgen05 | P2.4(S071--S093) | `cuda/m3_tcgen05/` |
| 4 | 完整 GEMM 四步走 | S084 + Session 4 | `cuda/m4_gemm/` |
| 5 | 低精度与 block scaling | P3(S094--S111) | `cuda/m5_lowprec/`、`kernels/` |
| 6 | TileLang 对照 | 收尾(S112) | (复用 assignment01) |
| Team | FlashKDA / MSA decode | — | `team/` |

# 环境与峰值

先确认工具链和卡都能发出 tensor core 指令：编译一个最小tensor core程序，测试卡的峰值性能，后面就可以用它来计算各个实现的性能达成率。

::: reading
Session3课件S008-S021；PTX ISA 的 mma 指令一节(形状与 dtype 表)；你手里
每块卡的官方 datasheet(或 whitepaper)。
:::

### 0.1 {.prob type=HANDS-ON file=cuda/m0_env/01_first_mma.cu}

编译并运行最小 Tensor Core 程序，它使用一个 warp 执行一条 m16n8k16 的 mma.sync 指令，并与 CPU 计算结果进行比较。

```
cd assignment02/cuda
make run/m0_env/01_first_mma
```

在你能使用的 GPU 上分别运行该程序（5090 使用 ARCH=120a make ...，B300 使用默认配置）。尝试使用不匹配的 ARCH 编译运行一次，记录现象，并结合 assignment01 Module 8 中 fatbin/JIT 的内容解释原因。可使用 make ptx/m0_env/01_first_mma 查看生成的 PTX。

### 0.2 {.prob type=DERIVE}

推导你所使用 GPU 的 Tensor Core 理论峰值。参考课上 A100 的推导方法（S018--S019），分别计算 5090 和 B300 的 bf16 峰值，并根据 dtype 宽度关系估算 fp8 / fp4 峰值。

开始计算前先明确采用的口径，包括 dense 或 sparse、boost 或 base 频率、FMA 是否计作 2 FLOP 等。完成推导后，再与 datasheet 中的官方数据进行对照；如果结果存在差异，需要说明差异来自哪一项口径。

| 量 | 5090 | B300 |
|---|---|---|
| bf16 FLOP/cycle/SM | | |
| bf16 峰值(TFLOPS) | | |
| fp8 峰值(TFLOPS) | | |
| fp4 峰值(TFLOPS) | | |
| datasheet 对照值与口径差异 | | |
| HBM/GDDR 带宽(GB/s) | | |
| 机器平衡点(FLOP/byte，bf16) | | |

根据 bf16 峰值和显存带宽计算机器平衡点（FLOP/byte），并与单条 mma 的计算强度（S016，m16n8k16 fp16 为 3.2 FLOP/byte）比较。思考两者之间的差距意味着什么，以及为什么后续 M2--M4 需要从数据供给路径入手优化。

### 0.3 {.prob type=CONCEPT}

判断下列说法是否正确，并给出一句理由。

(a) 一条 mma 的计算强度，分子是 $2MNK$，分母按 A、B 读入与 D 写回
的字节总和计(S016 的口径)。

(b) mma.sync 是 warp 级协作指令:32 个 lane 各持 fragment 的一部分，
要求全 warp 一致地执行这条指令；有 lane 发散时行为未定义。

(c) 增大 mma 的形状 M/N/K 能提高单条指令的计算强度，而且没有代价，
所以指令形状越大越好。

(d) 只要单条 mma 的计算强度低于机器平衡点，GEMM kernel 就不可能逼近
计算峰值。

# sm80:fragment 与 mma.sync

课上对 m16n8k16 fp16 推过 fragment 公式、手搓过单 tile mma
(C03--C07)。现在你手推一遍这个 m16n8k32 fp8 shape，
再看 ldmatrix 到底发挥了什么作用。

::: reading
课件 S025--S041、C03--C08；PTX ISA 的 "Matrix Fragments for
mma.m16n8k32" 一节与 "Warp-level matrix load instruction: ldmatrix"
一节；fp8 转换用 `cuda_fp8.h`(`__nv_fp8_e4m3`)。
:::

### 1.1 {.prob type=DERIVE file=cuda/m1_sm80/01_fragment_map.cu}

根据 PTX 文档中的 fragment 布局，推导
`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`
对应的 A、B fragment 映射公式，并完成文件中的四个函数。
判测使用纯 host 的 32-lane 真值表比对，无需 GPU 即可完成：

```
cd assignment02/cuda
make run/m1_sm80/01_fragment_map
```

附加问题（写入报告）：
A 的同一个 b32 寄存器中的 4 个 fp8 元素沿矩阵哪个方向相邻？
这个布局对 1.4 中使用 ldmatrix load 有什么影响？

A 的同一个 b32 中四个 FP8 元素具有相同的行坐标，并沿列/K 维
连续。由于 ldmatrix.b16 按原始 16 bit 搬运，可将每两个相邻
 FP8 视作一个 b16，使两个 b16 自然组成 MMA 所需的一个 b32；
 A 的 row-major shared-memory 布局与该装载方向一致。

### 1.2 {.prob type=DEBUG file=cuda/m1_sm80/02_bug_fragment.cu}

这个程序发一条 m16n8k16 fp16 mma，判测会 FAIL。先运行一遍，后改动:

(a) 描述症状：D 的哪些位置错、错成了什么(和对的部分是什么关系)；
    D 的下半块复制上半块。

(b) 修好它，并解释错的是哪个 fragment 的哪部分映射，为什么恰好产生(a)的症状。
    A fragment 的下半行映射缺少 +8 行偏移，导致上下半区使用相同的 A 数据。
    
```
cd assignment02/cuda
make run/m1_sm80/02_bug_fragment
```

::: {.capstone title="prob 1.3(FROM-SCRATCH):手写单 tile fp8 mma"}

在 `cuda/m1_sm80/03_mma_fp8.cu` 中从零实现一个单 tile 的 fp8 mma，不提供代码骨架。使用 `m16n8k32` e4m3 mma、f32 累加，并手动完成 fragment 装载（本题不使用 ldmatrix），最后与 CPU 参考结果进行严格相等比较。

要求如下：

输入使用小整数，保证转换为 e4m3 后可以精确表示；

程序接受一个 seed 参数，例如 `./prog 123`；

输出必须以 `PASS` 或 `MISMATCH` / `FAIL` 开头；

fp8 与 float 的互转直接使用 `cuda_fp8.h`。

1.1 中推导的 fragment 映射公式会直接用在这里。映射只要有一处错误，随机数据的判测就无法通过。

判测脚本会使用五个 seed，全部通过才算完成：

```
cd assignment02/cuda/m1_sm80
./judge_mma_fp8.sh 03_mma_fp8.cu
```

:::

### 1.4 {.prob type=MODIFY file=cuda/m1_sm80/04_ldmatrix.cu}

在给定骨架中保留 1.3 的手工装载方式，并另外实现一条使用 `ldmatrix` 的装载路径。两种实现需要共存，并分别通过判测。

根据 PTX 文档选择合适的 `ldmatrix` 变体（`.x1/.x2/.x4`，以及是否使用 `.trans`）。B 矩阵在 shared memory 中的布局还需要满足 `ldmatrix` 的 16 byte 行地址要求，具体约束见文件头说明。

两条路径都通过判测后，使用 `make ptx/m1_sm80/04_ldmatrix` 或 `nvdisasm` 查看生成的指令，分别统计 `smem → fragment` 阶段的装载指令和地址计算指令数量，并回答：

(a) `ldmatrix` 省掉了手工装载中的哪些工作？

(b) 为什么这些工作在手工装载路径中无法避免？

### 1.5 {.prob type=EXPERIMENT file=cuda/m1_sm80/05_ldsm_stride.cu}

观察行跨度对 `ldmatrix` 的影响。对同一个 16×16 fp16 tile，分别使用
32 B、64 B、128 B 和 128+16 B（padding）四种行跨度。

先根据课上介绍的 bank 模型，预测四种情况下执行一次 `ldmatrix`
所需的 wavefront 数及其比例，再运行程序并使用 Nsight Compute 测量：

```bash
cd assignment02/cuda
make run/m1_sm80/05_ldsm_stride
ncu --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./bin/m1_sm80/05_ldsm_stride
```

| 档位 | 预测 wavefront 比 | 实测 wavefront | 实测 conflict | 平均 cycle |
|---|---|---|---|---|
| 32 B |8|16384|8192|9.72|
| 64 B |16|32768|24576|10.75|
| 128 B |32|65536|57344|16.06|
| 128+16 B |4|8192|0|9.23|

比较预测与实测结果：哪一种行跨度使 wavefront 数增加到 4 倍？
wavefront 的比例应与 bank 模型一致，但实际耗时的差距通常没有这么大。
结合 8 个 warp 的占用情况，解释为什么 wavefront 增加 4 倍并不会使总耗时也增加 4 倍。

::: lookback

1.3--1.5 关注的是同一个问题：怎样把数据从 shared memory 装入
Tensor Core 的 fragment。1.3 手动完成每个 lane 的装载，1.4 使用
`ldmatrix` 简化这一步，而 1.5 说明即使使用 `ldmatrix`，shared memory
的 bank conflict 仍然会影响实际效率。

后面的模块会继续沿着这条数据供给路径展开：M2 中使用 descriptor
描述布局，M4 中进一步使用 TMA 和 pipeline 完成数据搬运。

:::

# smem 供数:descriptor 与 swizzle

sm90 起，tensor core 从 smem 取数不再经过 fragment 装载指令，而是
读一个 64 位 descriptor 按 canonical 布局自取。descriptor 与 swizzle
的格式在 sm100 的 tcgen05 上原样沿用，本模块推的每个公式都是 M3、
M4 的直接组件。

::: reading
课件 S042--S070(proxy:S051；core matrix 与 LBO/SBO:S057--S061；
swizzle:S062--S065)；PTX ISA 的 "Asynchronous Warpgroup MMA Shared
Memory Layout" 与 swizzling 小节。
:::

### 2.1 {.prob type=CONCEPT}

(a) 一个 warpgroup 使用 wgmma 读取刚写入 shared memory 的数据。将下面六个操作排成正确顺序，并说明每一步用于避免哪两个参与者之间的哪种乱序：

`wgmma.mma_async` / `st.shared` / `wgmma.commit_group` /
`fence.proxy.async` / `wgmma.fence` / `wgmma.wait_group`

(b) 判断下列说法是否正确，并给出一句理由。

1. `fence.proxy.async` 是 wgmma 专属的指令，TMA 与 tcgen05 的场景不需要它。
2. `wgmma.commit_group` 会阻塞，直到它之前发射的 wgmma 全部完成。
3. 不加 `fence.proxy.async` 时，wgmma 可能读到 shared memory 中的旧值，因为 `st.shared` 的写经过 generic proxy，而 wgmma 的读经过 async proxy。


### 2.2 {.prob type=DERIVE file=cuda/m2_smem/02_descriptor.cu}

根据文件头给出的位域，实现 SM100 的 64 位 smem matrix descriptor 编码函数，并分别对下面三种情况推导 LBO、SBO 和 layout：

- K-major，无 swizzle；
- K-major，128B swizzle；
- MN-major，128B swizzle。

判测为纯 host，无需 GPU。测试使用的描述符真值已经在 B300 上通过实际 tcgen05 指令验证，3.2 中也会使用这三组描述符：

```
cd assignment02/cuda
make run/m2_smem/02_descriptor
```

场景 2 和场景 3 最终得到的 descriptor 相同。在报告中回答：MN-major 与 K-major 的区别体现在哪里？


### 2.3 {.prob type=FROM-SCRATCH file=cuda/m2_smem/03_swizzle.cu}

实现 128B、64B 和 32B 三种 swizzle 模式的地址映射函数，将逻辑坐标映射到 atom 内的物理字节偏移。

课件 S064 已经推导了 128B swizzle 的地址位异或关系；64B 和 32B 两种模式需要根据 PTX ISA 的 swizzling 一节自行推导。

判测分为两部分：首先检查映射是否为双射，然后检查列访问是否存在 bank conflict。单纯使用 padding 或恒等映射虽然可能通过第一项，但无法通过第二项。判测同样为纯 host：

```
cd assignment02/cuda
make run/m2_smem/03_swizzle
```

本题实现的 `swizzle_128B` 会在 3.2 中直接用于 shared memory staging，并接受实际硬件上的 GEMM 判测。若 3.2 或 4.1 出现问题，可以先运行本题的判测，排除 swizzle 布局错误。


::: lookback

课件 C12--C14 给出了使用 wgmma 实现单 tile 的完整示例，本作业不单独设置对应题目。wgmma 是 sm_90a 专属指令，5090（sm120）和 B300（sm100）均无法运行；有兴趣的同学可以在集群 H 卡上自行尝试。

本模块重点保留 wgmma 路径中可以继续使用的两个部分：descriptor 与 swizzle。2.2、2.3 先分别完成它们的推导与判测，后面的 M3 会在实际硬件上继续使用。

:::

# sm100：tcgen05

在 sm100 上，tcgen05 将累加器放入 TMEM，并使用 mbarrier 处理异步完成通知。本模块会先在 B300 上完成一个单 tile GEMM，再通过后续实验理解 TMEM、mbarrier 和 2-CTA MMA 的使用方式。

::: reading
课件 S071--S093（TMEM：S073--S075；mma 与 idesc：S076--S079；
mbarrier：S080--S084；ld 与同步：S085--S086；七步流程：S087/F27；
2-CTA：S091）；PTX ISA 的 tcgen05 一族小节（alloc / mma / commit /
ld / fence）与 mbarrier 一节。
:::


### 3.1 {.prob type=CONCEPT}

判断下列说法是否正确，并给出一句理由。其中 (d) 需要写出计算过程。

(a) `tcgen05.ld` 读取 TMEM 时，每个 warp 只能读取自己对应的 32 条 lane，warp 之间不能互相读取。

(b) 与 `mma.sync` 由 warp 协作、wgmma 由 warpgroup 协作不同，`tcgen05.mma` 由单个线程发射，随后由硬件异步执行。

(c) TMEM 中的累加结果可以直接通过 TMA 搬回 global memory，不需要经过寄存器。

(d) TMEM 每个 SM 包含 128 lane × 512 column × 4 B；一个 m128n256 的 f32 accumulator 恰好占用其中一半。

(e) `tcgen05.commit` 会阻塞直到之前发射的 mma 全部完成，因此 commit 返回后即可安全读取 TMEM。


::: {.capstone title="prob 3.2(FROM-SCRATCH):tcgen05 单 tile GEMM" file=cuda/m3_tcgen05/02_single_tile.cu}

从零实现一个 tcgen05 单 tile GEMM：计算 m128n64k64 的 bf16 矩阵乘，使用 f32 累加、`cta_group::1`，并由一个 128 线程的 block 完成。

数据按照下面的路径流动：

`global → smem（K-major + 128B swizzle）→ tcgen05.mma → TMEM → tcgen05.ld → global`

代码骨架只保留课件 F27 中的七步流程注释。descriptor 使用 2.2 中实现的编码方式，shared memory staging 使用 2.3 中的 `swizzle_128B`。TMEM alloc、mbarrier、idesc 以及 mma / ld 的具体写法需要根据 PTX ISA 完成，相关语义可参考课件 C15--C21。

`fence.proxy.async` 的位置以及 `tcgen05.ld` 的 warp 可见范围已经在文件头中给出。

判测：

```
cd assignment02/cuda
make run/m3_tcgen05/02_single_tile
cd m3_tcgen05 && ./judge_tile.sh
```

在正确版本通过后，故意去掉 `fence.proxy.async` 再运行一次，记录出现的现象并写入报告。结合 2.1(a) 中的排序问题解释原因。

:::


### 3.3 {.prob type=DEBUG file=cuda/m3_tcgen05/03_bug_mbarrier.cu}

这个程序连续发射多轮 mma，并在每一轮后读取结果，用来模拟 M4 中沿 K 维流式计算的过程。当前程序的 mbarrier 使用存在错误，判测带有超时机制，因此程序挂死本身也是需要观察的现象。

先运行：

```
cd assignment02/cuda
make bin/m3_tcgen05/03_bug_mbarrier
cd m3_tcgen05 && ./judge_mbar.sh
```

(a) 分别记录 `rounds = 1`、`2`、`4` 时的运行结果以及问题出现的条件。

(b) 修复程序，并画出修改前后 mbarrier 的状态变化，包括 phase 和 arrival count。指出错误版本在哪一次等待中过早或过晚放行，并解释为什么会产生 (a) 中观察到的现象。

提示：注意 `tcgen05.ld` 与尚未完成的 mma 之间的关系。


### 3.4 {.prob type=EXPERIMENT file=cuda/m3_tcgen05/04_cta_pair.cu}

比较 `cta_group::1` 和 `cta_group::2` 完成同一个 m256n64k64 任务时的数据开销。

`cta_group::1` 使用两个独立 block；`cta_group::2` 使用一个 cluster 发射一条 M=256 的 mma，两个 CTA 分别保存 B 矩阵的一部分。

先预测，再运行实验：

(a) 在 `cta_group::2` 中，每个 CTA 所需的 B shared memory 是 `cta_group::1` 的多少？TMEM 的占用又如何变化？程序会打印 shared memory 用量，用它验证你的预测。

(b) 使用 Nsight Compute 比较两种实现的 shared memory 总流量。

(c) `cta_group::2` 节省下来的 shared memory 容量，在 M4 的 pipeline 中可以用来做什么？

(d) `cta_group::2` 依赖 sm90 引入的哪一种硬件机制？5090 不支持 2-CTA MMA，结合 0.2 中整理的硬件参数，说明为什么这类机制更常出现在数据中心 GPU 上。

运行：

```
cd assignment02/cuda
make run/m3_tcgen05/04_cta_pair
```

在这个单 tile 实验中，两种实现的运行时间差异处于噪声范围内，因此不要用耗时判断优劣。主要比较 shared memory 用量以及 Nsight Compute 测得的流量。

# 完整 GEMM

本模块将 3.2 的单 tile 实现扩展为完整的 B300 GEMM，并依次加入
tiling、TMA 和多级 pipeline。实验统一使用 4096³ 的 bf16 GEMM，
tile 大小固定为 128×64×64，并与 cuBLAS 结果进行严格相等比较。

::: reading
课件 S084（pipeline 骨架）、S087；Session 4 讲义对应章节；PTX ISA
的 `cp.async.bulk.tensor`、mbarrier（`expect_tx`）小节；CUDA Driver
API 的 `cuTensorMapEncodeTiled`。
:::

下面的表格贯穿 4.1--4.3。第 0 行需要在同一块 GPU 上重新运行
assignment01 Bonus 中的 naive matmul。由于该实现使用 fp32，只比较性能量级。

| 实现 | TFLOPS | 对 cuBLAS 达成率 | 一句话：时间主要花在哪 |
|---|---|---|---|
| naive（assignment01，fp32） | | | |
| 4.1 tiled |29.7 TFLOPS|(cuBLAS 1057.2, 达成率 3%)|staging、同步、MMA、等待完全串行，没有重叠, 每个 K tile 都有 __syncthreads、commit 和 mbarrier wait|
| 4.2 TMA |577.0 TFLOPS|(cuBLAS 1767.2, 达成率 33%)|单缓冲造成的 TMA→MMA 串行|
| 4.3 pipeline（S=3） |602.4 TFLOPS |(cuBLAS 1767.0, 达成率 34%) |smem大小不够导致pipline和SM多block驻留并发的tradeoff|
| cuBLAS | | 100% | |


### 4.1 {.prob type=FROM-SCRATCH file=cuda/m4_gemm/01_tiled.cu}

将 3.2 的单 tile 实现扩展为完整 GEMM：使用 grid 覆盖所有输出 tile，
并沿 K 维循环完成累加。数据 staging 仍然使用 `st.shared` 和 swizzle。

相对 3.2，本题主要新增 tile 映射以及 K 循环中的同步和累加。
具体步骤见文件头。

```
cd assignment02/cuda
make run/m4_gemm/01_tiled
```

通过判测后，填写性能表中 `4.1 tiled` 一行。结合 0.2 中计算的机器
平衡点，判断此时性能主要受哪一环节限制。


### 4.2 {.prob type=MODIFY file=cuda/m4_gemm/02_tma.cu}

将 4.1 中的 shared memory staging 改为 TMA。

host 端使用 `cuTensorMapEncodeTiled` 创建 tensor map；kernel 中使用
`cp.async.bulk.tensor` 搭配 mbarrier 的 `expect_tx` 完成搬运。
本题仍然使用单缓冲。

tensor map 的参数以及同步方式的变化已经在文件头中给出。
swizzle 由 TMA 根据 tensor map 自动完成，原有 descriptor 不需要修改字段。

```
cd assignment02/cuda
make run/m4_gemm/02_tma
```

通过判测后，填写性能表中 `4.2 TMA` 一行。

比较 4.1 与 4.2，并回答：4.1 中 shared memory staging 的开销由哪些
部分组成？改用 TMA 后，其中哪些工作不再由普通 CUDA 指令完成？

使用 Nsight Compute 辅助分析，观察 4.1 中 SM 时间主要消耗在哪些部分。

下列工作不再由普通 CUDA 线程逐元素完成：
- global load 的地址生成与事务组织。
- 逐元素 shared-memory store。
- 线程内 staging 循环及循环控制。
- 软件计算 swz128 地址。
- 由全部 128 个线程协作完成搬运。
- generic → async proxy 所需的 fence.proxy.async。
现在只由一个线程发射两条 TMA，硬件根据 tensor map 自动完成全局地址遍历、成块搬运、128B swizzle 和 shared-memory 写入，并通过 full mbarrier 报告完成。
仍然存在的工作包括：
- TMA 发射和 full mbarrier 等待。
- tcgen05 MMA、commit 和 empty mbarrier。
- TMEM epilogue。
- 单缓冲造成的 TMA→MMA 串行；4.3 pipeline 才会进一步重叠二者。

::: {.capstone title="prob 4.3(FROM-SCRATCH):多级流水" file=cuda/m4_gemm/03_pipeline.cu}

将 4.2 的单缓冲改为 `STAGES` 级循环缓冲，使后续 K tile 的 TMA
预取能够与当前 tile 的 mma 计算重叠。

`STAGES` 为编译参数，例如：

```
STAGES=4 make -B run/m4_gemm/03_pipeline
```

这里的 `-B` 不能省略，否则修改 `STAGES` 后可能不会重新编译。

文件头给出了每个 stage 使用双 mbarrier 的基本结构，同时包含一个需要
处理的 pipeline hazard：强制发射与机会式预取之间的边界处理错误会导致
死锁。在该错误下，1024³ 可能偶尔通过，而 4096³ 会稳定暴露问题。
开始实现前先阅读文件头说明。

本题不要求 warp specialization、persistent kernel 或 epilogue 融合，
也不设置 cuBLAS 达成率门槛。重点是流水实现是否正确，以及能否根据实验
结果解释性能变化。

```
cd assignment02/cuda
make run/m4_gemm/03_pipeline
cd m4_gemm && ./sweep_stages.sh
```

完成以下内容：

1. 使用 `S=3` 的结果填写性能表中 `4.3 pipeline` 一行。

2. 完成不同 stage 数的性能测试：

   | 形状 | S=2 | S=3 | S=4 | S=6 |
   |---|---|---|---|---|
   | 4096³ |584.3 TFLOPS|601.8 TFLOPS|509.1 TFLOPS|308.3 TFLOPS|
   | 256 × 4096 × 16384 |233.0 TFLOPS|300.8 TFLOPS|310.5 TFLOPS|305.1 TFLOPS|

   比较两个形状对 `STAGES` 的敏感程度，并结合 shared memory 用量、
   每个 SM 可同时驻留的 block 数以及 block 间并发能够隐藏的延迟进行解释。

   大 grid 本身能够利用多 block/SM 隐藏延迟，所以过深 pipeline 造成的 occupancy 损失更突出；小 grid、长 K 缺少 block 间并发，因而更依赖块内多级流水，最优 stage 数更深，并且在 S=4 左右趋于饱和。

3. 任选一个 `STAGES`，画出稳态阶段各 stage 中 TMA 与 mma 的流水时空图。

4. 回答：

   (a) 从 4.1 到 4.3，主要瓶颈发生了哪些变化？
   4.1：瓶颈主要在软件 staging 和同步，4.2：瓶颈移动到单缓冲串行依赖，4.3：减少暴露的 TMA 延迟，瓶颈进一步移向 MMA/TMA 吞吐和资源并发
   (b) 梯子表中每一级优化分别减少了哪部分开销？
  | 优化 | 主要减少的开销 | 没有减少的部分 |
|---|---|---|
| naive → 4.1 tiled | 通过 tiling 增加 A/B 数据复用；使用 Tensor Core 替代大量普通标量 FMA；降低每 FLOP 对应的 global-memory 流量 | 软件 staging、swizzle、同步仍很重 |
| 4.1 → 4.2 TMA | 普通线程的 global 地址生成、事务组织、逐元素 shared store、staging 循环、软件 swizzle；不再需要 generic→async proxy fence | global 搬运字节数没有消失；TMA 与 MMA仍串行 |
| 4.2 → 4.3 pipeline | 不减少 TMA 字节数或 MMA 数量，而是通过重叠隐藏原本暴露在关键路径上的 TMA/full-wait 延迟 | 增加 shared-memory 容量；仍有 barrier、MMA commit、epilogue |
   (c) 如果继续增大 tile 或增加 stage 数，shared memory 与 TMEM
   哪一个会先成为容量限制？结合 3.4(c) 的结果说明。
   增加 stage：SMEM 增长，TMEM不变，SMEM必然先限制。
   扩大 tile：SMEM和TMEM都可能增长；
   但在当前S=3及其以上的设计中，SMEM已经明显压低block/SM，
   因此通常先成为实际限制。
:::


### 4.4 {.prob type=FROM-SCRATCH opt=Optional}

任选一个方向继续优化：

(a) 实现 4.3 的 `cta_group::2` 版本，利用 B shared memory 用量的减少
增加 pipeline 深度或扩大 tile；

(b) 在 5090 上使用 `mma.sync` 实现相同的 tiling，并与 B300 的结果比较；

(c) 自由优化当前 kernel，提高相对 cuBLAS 的性能，并记录每一步优化
解决了什么问题。


### 4.5 {.prob type=EXPERIMENT file=cuda/m4_gemm/05_thin_gemm.cu}

观察矩阵形状较窄时 Tensor Core GEMM 的性能变化。

实验形状取自 vLLM 主树中 Kimi K3 的 decode GEMM dispatch 表：

`vllm/models/kimi_k3/nvidia/low_latency_gemm.py`

使用其中七个投影层对应的 N、K，并对 M 取：

$\{1, 8, 16, 64, 256, 1024, 4096, 16384, 65536\}$

其中 N 和 K 由权重形状决定，变化的 M 表示一次 GEMM 处理的 token 数量。
较小的 M 对应 decode batch；较大的 M 可以对应 chunked prefill 中单次
处理的 chunk。vLLM 在 $M \le 16$ 时会放弃 cuBLAS / Tensor Core 路径，
改用 CUDA Core FMA 的 skinny kernel。

运行程序前，先对每个形状计算 arithmetic intensity：

$$
AI = \frac{2MNK}{2MK + 2NK + 2MN}
$$

将结果与 0.2 中得到的机器平衡点比较，并计算对应的理论性能上限：

- compute bound：Tensor Core 峰值；
- memory bound：$AI \times$ 显存带宽。

然后运行：

```
cd assignment02/cuda
make bin/m4_gemm/05_thin_gemm
./bin/m4_gemm/05_thin_gemm <峰值TFLOPS> <带宽GB/s>
```

其中两个参数使用 0.2 中得到的数值。

程序会输出 TFLOPS、GB/s、AI 以及相对于 compute roof 和 memory roof
的达成率。根据结果回答：

(a) 描述性能随 M 变化的趋势。M 较小时，Tensor Core 性能从什么时候
开始明显下降？M 增大到什么范围后进入相对稳定的平台？平台上的达成率是多少？

(b) 对性能下降的形状，比较 compute roof 和 memory roof 两种达成率。
哪些形状主要受到显存带宽限制？

(c) `f_b_proj`（K=128）中两个 roof 的达成率都较低。结合矩阵形状分析，
它的限制来自哪里？

(d) 根据上述结果解释 vLLM 在 $M \le 16$ 时选择 skinny CUDA Core
kernel 的原因。

结论很清楚：六个常规形状在 M=4096–65536 进入平台，平均约达到 Tensor Core 峰值的 79%；从 M=1024 开始下滑，M≤256 明显塌陷。f_b_proj 因 K=128 属于另一种机制，需要单独讨论。

(a) 性能随 M 的趋势

随着 M 增大，TFLOPS 快速上升；原因是 arithmetic intensity 和可并行 CTA 数量增加，同时 kernel 启动、调度及 Tensor Core setup 等固定成本被摊薄。
从大 M 向小 M 看：
- M≥4096：六个常规形状进入相对稳定平台。
- M=1024：开始偏离平台，六层平均 %TCpeak 从约 79% 降至 62.3%，属于过渡区。
- M=256：平均只剩 39.2%，已经明显塌陷。
- M=64：平均 12.2%。
- M=16：平均仅 3.3%。
六个常规形状在 M≥4096 的平台数据为：

| 形状 | 平台 TFLOPS 范围 | 平均 `%TCpeak` |
|---|---|---|
| `q_b_proj` | 1755.6–1978.1 | 83.2% |
| `o_proj` | 1885.2–2013.7 | 87.1% |
| `fused_qkv_a_proj` | 1638.3–1761.9 | 76.3% |
| `in_proj_qkvgfab` | 1684.9–1761.5 | 76.5% |
| `dense_down_proj` | 1626.9–1839.7 | 75.7% |
| `dense_gate_up_proj` | 1653.6–1730.7 | 75.1% |


因此可以概括为：平台约 1.63–2.01 PFLOPS，达成率约 72%–90%，六层总体平均约 79%。
f_b_proj 没有进入相同的 compute 平台：其 TFLOPS 从 M=4096 的 325 继续增长到 M=65536 的 692，%TCpeak 也只有 14.4%–30.8%。

(b) compute roof 与 memory roof 的比较

在理论 roofline 分类上，M≤256 时所有形状的 AI 都低于机器平衡点 281.25 FLOP/byte，因此 memory roof 都低于 compute roof。但是否真的主要受 HBM 带宽限制，还要看 %BW 是否足够高。

| 形状 | M≤16 的 `%TCpeak` | M≤16 的 `%BW` | 判断 |
|---|---|---|---|
| `dense_gate_up_proj` | 0.3%–4.4% | 75.8%–79.8% | 明显带宽侧限制 |
| `in_proj_qkvgfab` | 0.2%–3.8% | 66.1%–71.9% | 主要带宽侧限制 |
| `o_proj` | 0.2%–4.0% | 57.4%–74.1% | 主要带宽侧限制 |
| `dense_down_proj` | 0.2%–3.5% | 58.9%–63.9% | 主要带宽侧限制 |
| `fused_qkv_a_proj` | 0.1%–2.7% | 39.8%–48.0% | 带宽与固定开销混合 |
| `q_b_proj` | 0%–1.4% | 9.5%–24.7% | 两个 roof 都低，非纯带宽限制 |
| `f_b_proj` | 0%–0.1% | 1.3%–1.4% | 形状、并行度和固定延迟主导 |


所以，最明确受显存带宽侧限制的是：
- dense_gate_up_proj
- in_proj_qkvgfab
- o_proj
- dense_down_proj
fused_qkv_a_proj 有明显带宽影响，但并未充分接近 memory roof。q_b_proj 和小 M 的 f_b_proj 连带宽 roof 都离得很远，不能简单解释为“HBM 已打满”。
需要注意：这里的 GB/s 是按 A、W、D 各搬运一次计算的有效带宽，不是 profiler 直接测得的 HBM transaction；重复访问还可能命中 L2。因此严格确认 HBM 瓶颈仍需 Nsight Compute。
(c) f_b_proj 为什么两个 roof 都低

f_b_proj 的形状为 N=1536, K=128。它有三个相互叠加的问题：

1. 理论 AI 上限很低

   当 $M \to \infty$ 时：
$$
AI_{\max}
= \frac{NK}{N+K}
= \frac{1536 \times 128}{1536 + 128}
\approx 118.15
$$
   这远低于 B300 SXM 的机器平衡点 281.25，因此它无论 M 多大都无法转成 compute-bound。

2. K=128 的 reduction 太短
   每个输出 tile 只需要很少的 MMA K 迭代，Tensor Core 计算时间不足以覆盖数据搬运、同步、调度和 epilogue，pipeline 也难以充分展开。

3. 小 M 时并行度和传输规模都太小
   N=1536 能提供的 N tile 数有限，小 M 又只能提供一两个 M tile，无法生成足够 CTA 填满 B300 的 SM。其小矩阵耗时长期维持在约 4 μs，说明固定延迟占主导；同时数据量又不足以让 HBM 达到峰值，所以 %TCpeak 和 %BW 都低。


随着 M 增大，并行度改善，%BW 最终升到 73.4%；但受 AI 上限限制，%TCpeak 在最大 M 也只有 30.8%。

(d) 为什么 M≤16 选择 skinny CUDA Core kernel

  M≤16 时，所有形状的 %TCpeak 只有 0%–4.4%，而同一层从 M=1 增长到 M=16 时，kernel 时间通常几乎不变。这说明运行时间主要不是 Tensor Core 计算，而是：
- kernel launch 和 cuBLAS dispatch；
- Tensor Core/TMA/tile setup；
- 同步和 epilogue；
- CTA 数量不足；
- tile 填充和尾部浪费；
- 短 kernel 无法隐藏访存延迟。
  skinny CUDA Core kernel 直接针对少量输出行安排 FMA，绕过较重的 Tensor Core 数据供给与调度路径。虽然 CUDA Core 的理论峰值较低，但在这个区间真正重要的是固定延迟而不是峰值 FLOPS，因此端到端 decode 延迟反而更低。
  本次计算使用 B300 SXM dense BF16 2250 TFLOPS 和 8000 GB/s；官方资料给出的 B300 SXM 带宽为最高 8 TB/s，B300 SXM 规格，BF16 dense 为官方 sparse 数值的一半，HGX B300 规格。

`in_proj_qkvgfab` 对应 KDA 的输入投影。完成团队题 C1/C2 的同学可以
使用这一行的实验结果作为后续分析的参考。

```
layer                    M      N      K        us    TFLOPS      GB/s      AI  %TCpeak      %BW
in_proj_qkvgfab          1   6288   7168      17.1       5.3    5285.6     1.0     0.2%    66.1%
in_proj_qkvgfab          8   6288   7168      15.7      45.9    5750.4     8.0     2.0%    71.9%
in_proj_qkvgfab         16   6288   7168      16.7      86.3    5421.3    15.9     3.8%    67.8%
in_proj_qkvgfab         64   6288   7168      15.1     381.7    6078.0    62.8    17.0%    76.0%
in_proj_qkvgfab        256   6288   7168      22.7    1015.6    4270.3   237.8    45.1%    53.4%
in_proj_qkvgfab       1024   6288   7168      61.9    1491.4    1901.7   784.2    66.3%    23.8%
in_proj_qkvgfab       4096   6288   7168     215.0    1717.2     931.9  1842.7    76.3%    11.6%
in_proj_qkvgfab      16384   6288   7168     838.4    1761.5     633.4  2781.0    78.3%     7.9%
in_proj_qkvgfab      65536   6288   7168    3506.2    1684.9     528.7  3186.7    74.9%     6.6%
```

::: lookback

4.1--4.3 从同一个 GEMM 出发，依次加入 tiling、TMA 和多级 pipeline。
回顾梯子表时，重点不是最终的 cuBLAS 达成率，而是能够说明每一步优化
减少了什么开销，以及新的瓶颈出现在哪里。

assignment01 Bonus 中 naive matmul 与 cuBLAS 之间的部分差距，现在已经
可以从 Tensor Core、异步数据搬运和软件流水三个方面解释。进一步的
warp specialization、更大的 tile、epilogue 融合和 persistent kernel
等优化不在本作业要求范围内。

4.5 则说明 Tensor Core 的收益同样依赖矩阵形状。当 M 很小时，即使
kernel 本身使用了前面的优化方法，也不一定能够有效利用 Tensor Core。

:::

# 低精度与 block scaling

降低数值精度可以换取更高的 Tensor Core 吞吐，但代价是可表示的动态范围变小。本模块先从 per-tensor scale 遇到 outlier 时的问题入手，再分析 block scaling 的代数约束，最后完成一条 NVFP4 量化通路，并通过融合实验观察低精度 kernel 的实际性能。

::: reading
课件 S094--S111（fp8 格式：S096；per-tensor 与 outlier：S097--S099；
block scaling 约束：S100--S101；fp4 与 NVFP4：S106--S108；SF 布局：
S108）；`cuda_fp4.h` 中的 `__nv_fp4x2_e2m1`；cuBLASLt 的 block-scaled
FP4 matmul。格式约定见题面材料 `cuda/m5_lowprec/nvfp4_common.h`。
:::


### 5.1 {.prob type=EXPERIMENT file=kernels/quant_outlier.py}

观察 outlier 对 per-tensor scale 的影响。输入包含一万个 $[-1,1]$
均匀分布的元素，并额外加入一个值为 3000 的 outlier。补全脚本中的两个
TODO，完成 per-tensor E4M3 量化与反量化：

```
cd assignment02
uv run python kernels/quant_outlier.py
```

| 采样点 $x\approx$ | 0.5 | 0.1 | 0.01 | 0.005 | 3000 |
|---|---|---|---|---|---|
| 相对误差 | 4.611e-02 | 4.634e-02 | 3.085e-01 | 1.000e+00 | 0.000e+00 |

根据实验结果回答：

(a) 去掉 outlier 后重新量化，$x\approx0.5$ 处的误差变化多少倍？

149.42x

(b) 找出输入被量化为 0 的阈值，并写出该阈值与 scale 的关系式。

$$
x_{threshold} = scale * 2 ^ {-10} \approx 0.006539
$$

(c) 改用 1×128 的 per-block scale 后，包含 outlier 的 block 与不包含
outlier 的 block 分别有什么变化？

不含 outlier 的 block 使用较小的 scale，普通数值保持较高精度。

含 outlier 的 block 仍被 3000 控制，较小元素误差变大，小于约 0.006539 的元素会量化成 0。

其他 block 不受 outlier 影响，所以 per-block scaling 把 outlier 的影响限制在了一个 block 内。

### 5.2 {.prob type=DERIVE file=kernels/block_scale_sim.py}

block scaling 的代数关键是沿着哪个方向分段 scale 乘积在
K 归约中才能保持常数。补全两个 fp64 模拟函数:

- `gemm_scale_per_row_col`:A 每行一个 scale，B 每个输出列一个
  scale。scale 乘积在整个点积中不变，完整归约后只用乘回一次。
- `gemm_scale_along_k`:A 和 B 的 scale 每 128 个 K 元素改变
  一次。每个 K block 的 partial sum 要分别乘回该段的 scale 乘积，
  然后进行累加。

文件还提供了错误范例 `gemm_scale_along_k_one_restore`：它先把
所有归一化 partial sum 相加，最后只乘回第一段的 scale。

判测:

```
cd assignment02
uv run pytest tests/test_block_scale.py
```

两个正确函数只要在 fp64 容差内与直接 GEMM 等价，不要要求
bit-exact（因为分段会改变浮点加法的分组）。然后回答:

(a) 用两行代数式分别写出 row/column scale 为什么可以在整个
点积外乘回，而 K-block scale 为什么必须逐段乘回。

设归一化后的值为 $qA$、$qB$，且代码计算 $C=AB^T$。

Row/column scale 在整个 K 维不变，因此可以提出求和：

$$
C_{ij}=\sum_k(s^A_iqA_{ik})(s^B_jqB_{jk})
=s^A_i s^B_j\sum_k qA_{ik}qB_{jk}.
$$

K-block scale 随分段 $g$ 改变，只能对每段分别乘回：

$$
C_{ij}=\sum_g\sum_{k\in K_g}(s^A_{ig}qA_{ik})(s^B_{jg}qB_{jk})
=\sum_g s^A_{ig}s^B_{jg}\left(\sum_{k\in K_g}qA_{ik}qB_{jk}\right).
$$

因为 $s^A_{ig}s^B_{jg}$ 随 $g$ 变化，所以不存在一个公共 scale 能从整个 K 求和中提出。

(b) [NVIDIA CUTLASS 的 Blackwell SM100 GEMM 说明](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)
把 A 的
scale 布局写成 $M \times \lceil K/SV \rceil$，B 写成
$N \times \lceil K/SV \rceil$，每个 scale 负责连续的 16 或 32 个 K
元素。结合 GEMM 内循环、连续供数和 Tensor Core 指令语义，说明硬件
为什么把 block scale 与 K 归约对齐。

(c) DeepSeek-V3 的 weight 128×128 表示 scale 还在输出通道方向上
共享，但每个组仍覆盖一段 K；NVFP4 则每 16 个 K 元素一组。
结合 5.1(c) 的误差，说明粒度 16 相对粒度 128 有什么优势，又增加了
多少 scale metadata 与供数复杂度。

### 5.3 {.prob type=FROM-SCRATCH file=cuda/m5_lowprec/}

完成一条 NVFP4 量化通路。开始前先阅读 `nvfp4_common.h` 中给出的格式约定：

- 每个量化组包含 K 方向连续的 16 个元素；
- SF 使用 `amax / 6`，并以 e4m3 保存；
- SF 按 Tensor Core 消费要求的 swizzled 布局存放，字节偏移公式已经给出。

#### (a) E2M1 编码

在 `e2m1_encode.h` 中实现 host/device 通用的 E2M1
round-to-nearest-even 编码器。

判测会在 GPU 上使用 `cuda_fp4.h` 中的 `__nv_fp4x2_e2m1`
转换同一批候选值，包括所有中点、边界以及采样值，并与自己的编码结果
逐位比较：

```
cd assignment02/cuda
make run/m5_lowprec/03a_encode_check
```

#### (b) NVFP4 quant kernel

在 `nvfp4_quant_kernel.h` 中实现 quant kernel，完成

`bf16 → e2m1 打包 + e4m3 SF + swizzled SF 布局`

设备侧的 E2M1 转换直接使用 `__nv_fp4x2_e2m1`。

第一层判测要求输出逐 byte 与 host 参考结果相等；host 参考使用你在
5.3(a) 中实现的 E2M1 编码器生成。随后将量化结果原样交给 cuBLASLt
的 FP4 matmul，检查生成的数据和 SF 布局能否被实际消费：

```
make run/m5_lowprec/03b_nvfp4_quant
make run/m5_lowprec/test_fp4_gemm
```

正确实现下，FP4 matmul 的 `maxrel` 约为 `4e-3`，主要来自 bf16
输出的舍入误差。如果 SF 布局错位或 scale 对应到了错误的量化组，
通常会出现成块的明显数值错误，而不是仅有小幅舍入误差。

#### (c) Ceiling probe

在 `03c_ceiling_probe.cu` 中实现一个 ceiling probe。它与 quant kernel
使用相同的访存模式，但不执行量化计算，只读取数据并通过简单 xor 后写回。

报告：

- ceiling probe 的 GB/s；
```
./bin/m5_lowprec/03c_ceiling_probe
M=4096   K=7168   probe    22.59 us    3330 GB/s
M=16384  K=4096   probe    45.04 us    3818 GB/s
M=16384  K=8192   probe    84.30 us    4080 GB/s
```
- quant kernel 的 GB/s；
```
./bin/m5_lowprec/03b_nvfp4_quant
M=128   K=1024   PASS(bad=0)      6.16 us      55 GB/s
M=200   K=4096   PASS(bad=0)      6.16 us     341 GB/s
M=4096  K=7168   PASS(bad=0)     65.56 us    1148 GB/s
```
- 两者的比值。

结合 Nsight Compute 判断 quant kernel 距离自己的访存上限还有多远，
以及剩余差距主要来自访存还是计算。

#### (d) Optional

使用 tcgen05 `kind::mxf4nvf4` 直接消费自己生成的 NVFP4 数据，
SF 通过 TMEM 提供，并与 cuBLASLt 的结果对拍。


::: lookback

5.3(c) 用来区分量化 kernel 中的访存成本和数值转换成本。作为参考，
软件实现 E2M1 RN-even 时，每个 kernel 约执行 18129 条 FSETP；
使用硬件转换后约为 1272 条 `F2FP.E2M1`，同一 kernel 的时间从
71.7 µs 降到 32.8 µs。Nsight Compute 中的瓶颈也从 SM 利用率
84.7% 的 compute-bound 转为 memory-bound。

这组结果说明，当低精度转换由专用硬件完成后，量化 kernel 可以更接近
纯数据搬运的性能特征；5.3(c) 的 ceiling probe 就用于衡量这一差距。

Hopper 上的 fp8 累加需要软件 promotion（S102--S105，DeepGEMM 使用
双层累加循环），而 Blackwell 进一步在硬件中支持 block scaling
（S105）。本作业不单独设置对应题目，但 5.2 中的 scale 分组约束是
理解两者的共同基础。

:::


::: {.capstone title="prob 5.4(FROM-SCRATCH):融合 rms_norm + NVFP4"}

实现融合的 `rms_norm + NVFP4` kernel：

$$
y = \mathrm{rms\_norm}(x)\cdot w
$$

要求将结果直接量化为 NVFP4，不写回 bf16 中间结果。

本题以 vLLM fused kernel 清单中的 issue #25179，以及两个相关实现
PR #36413、#32957 为背景。相关实现的正确性没有问题，也确实将 kernel
数量从 2 个减少到了 1 个，但当时观察到的端到端收益仍处于 ±1% 左右的
测量噪声内。本题要通过逐形状实验解释这部分收益去了哪里。

按理论字节量计算，两步实现约为 6.56 B/elem，融合后约为
2.56 B/elem，因此单纯从访存量估计可以得到约 2.56× 的理想加速。
这个数字只是上限，实际结果需要通过实验验证。

程序已经提供两步基线、正确性检查和逐形状计时框架。测试覆盖：

- $M$ 从 1 到 16384；
- $K \in \{4096, 7168, 8192\}$；
- 共十个测试形状。

需要完成三部分：

1. 实现融合 kernel，具体线程组织和 tile 结构不限；
2. 分别调优融合实现和两步基线，使两边都使用各自合理的配置后再比较；
3. 完成逐形状性能分析。

第二点必须认真处理。让两步基线处于明显不利的配置下得到的加速比没有
比较意义，这也是相关上游 PR 的实验过程中暴露出的一个问题。

运行：

```
cd assignment02/cuda
make run/m5_lowprec/04_fused_rms_nvfp4
```

报告逐形状结果，包括：

- 实测加速比；
- 融合 kernel 相对 5.3(c) ceiling probe 的性能比例。

重点解释实测结果与 2.56× 理论上限之间的差距：不同 M 范围内主要受到
什么因素限制，并给出 Nsight Compute 数据或计算结果作为依据。

本题不设置性能门槛。实现首先需要保证正确，评分重点放在实验设计和性能归因。

# 5.4 融合 RMSNorm + NVFP4 Quant 性能分析报告

## 1. 实验目标

本题实现融合的 RMSNorm + NVFP4 quant kernel。计算语义为

$$
rnorm=\frac{1}{\sqrt{\frac{1}{K}\sum_{i=0}^{K-1}x_i^2+\epsilon}},
$$

$$
y_i=x_i\cdot rnorm\cdot w_i,
$$

随后按照每 16 个元素一组，将 $y$ 直接量化为 NVFP4。与两步实现不同，融合 kernel 不将 RMSNorm 的结果以 bf16 中间张量写回显存。

两步实现的理论访存量约为

$$
6.56\ \mathrm{B/elem},
$$

融合后约为

$$
2.56\ \mathrm{B/elem},
$$

因此，如果运行时间完全由显存传输量决定，理想加速比为

$$
\frac{6.56}{2.56}\approx 2.56\times.
$$

实验的主要目的并不是验证融合是否正确，而是分析为什么实际性能提升明显低于这一理论上限。

---

## 2. Kernel 实现

融合 kernel 采用 block-per-row 的结构，并让不同 block 独立处理不同 row。

RMSNorm 阶段中，每个线程使用 `float4` 一次读取连续 8 个 bf16，即一次进行 16 B 的向量化加载。线程首先累积自己的平方和，然后使用 warp shuffle 完成 warp 内归约，各 warp 的结果写入 shared memory，再由第一个 warp 完成 block 级归约并计算 `rnorm`。

计算出 `rnorm` 后，同一 block 继续对该 row 进行 NVFP4 quant。每个线程负责若干个 16-element group。对每组数据，使用两个 `float4` 分别加载 16 个 bf16 输入和对应的 weight，计算

$$
y_i=x_i\cdot rnorm\cdot w_i,
$$

并将 16 个 FP32 中间结果暂存在寄存器中，同时计算该组的 `amax`。之后按照 5.3(b) 的规则计算 E4M3 scale：

$$
sf8=\mathrm{E4M3}\left(\frac{amax}{6}\right),
$$

再通过 `__nv_fp4x2_e2m1` 将两个 FP32 一次转换并打包为两个 E2M1 FP4。

因此整个数据流为

```text
bf16 x
   ↓
RMS reduction
   ↓
rnorm
   ↓
x * rnorm * w
   ↓
amax + E4M3 scale
   ↓
E2M1 FP4
```

中间的 RMSNorm 输出始终保留在 kernel 内，不产生 bf16 中间张量。

正确性测试中所有十个 shape 均通过允许 $10^{-4}$ 比例 byte mismatch 的判测。

---

## 3. 公平调参与实验设计

一个重要发现是，如果只优化 fused kernel，而保持题目默认的 baseline 配置，会严重高估融合收益。

### 3.1 Grid 大小

最初 fused kernel 使用：

```cpp
grid = min(M, 2 * sms);
```

这种写法本质上让少量 block 以 grid-stride 的方式持续处理多行。

实测发现这种 persistent-style 配置明显不适合本 kernel。例如 `BLOCK=256, M=16384, K=4096`：

| Grid | Fused 时间 |
|---|---:|
| $1\times SM$ | 330.34 μs |
| $2\times SM$ | 185.55 μs |
| $4\times SM$ | 122.98 μs |
| $M$ | **106.81 μs** |

随着可调度 block 数增加，性能持续提升，因此最终采用 one-block-per-row：

```cpp
grid = M;
```

这说明本 kernel 中 reduction、同步、量化和数据访问具有较长的单-row 执行链，提供更多独立 row block 可以显著提高 GPU 调度自由度和 latency hiding 能力。

### 3.2 Block 大小

进一步测试 `BLOCK=128/256/512`。

对于 fused kernel，大多数中大规模 shape 上 `BLOCK=128` 最优。例如：

$$
M=16384,\ K=4096
$$

时：

```text
BLOCK=512 : 141.19 us
BLOCK=256 : 106.81 us
BLOCK=128 :  84.07 us
```

较小 block 能提供更细粒度的 block-level parallelism。尽管每个线程需要处理更多元素或更多 quant group，但大量独立 row 可以同时参与调度，总吞吐反而更高。

`M=256,K=4096` 是少数 `BLOCK=256` 略优的 shape，因此最终按逐 shape 实测选择 fused block 配置。

### 3.3 两步 baseline

为了保证比较公平，baseline 同样独立调优。

RMSNorm baseline 在大 M 下通常以 `BLOCK=128, grid=M` 更快；小 M 下由于缺乏 row-level parallelism，更大的 BLOCK 能提高单行内部并行度，因此部分 shape 使用 256 或 512。

Standalone NVFP4 quant 对 `BLOCK=128/256/512` 进行了比较，128 在大多数 shape 上最优，因此最终采用：

```cpp
BLOCK = 128;
grid = ceil(numGroups / BLOCK);
```

这一调参非常重要。未优化 baseline 时，大规模 shape 曾表现出接近 $2\times$ 的融合加速；baseline 调优后，加速下降到约 $1.3\sim1.6\times$。这说明如果 baseline 本身处于明显不利的配置，得到的高加速比并不能反映 fusion 的真实收益。

---

## 4. 最终性能结果

最终优化后的结果为：

| M | K | Two-step | Fused | Speedup | 2.56×上限达成率 |
|---:|---:|---:|---:|---:|---:|
| 1 | 4096 | 8.21 μs | 6.16 μs | 1.33× | 52.1% |
| 16 | 4096 | 9.23 μs | 6.16 μs | 1.50× | 58.5% |
| 256 | 4096 | 11.29 μs | 6.33 μs | **1.78×** | 69.7% |
| 1024 | 4096 | 16.41 μs | 10.26 μs | 1.60× | 62.5% |
| 4096 | 4096 | 38.58 μs | 24.96 μs | 1.55× | 60.4% |
| 16384 | 4096 | 121.60 μs | 84.05 μs | 1.45× | 56.5% |
| 4096 | 7168 | 61.52 μs | 41.04 μs | 1.50× | 58.6% |
| 16384 | 7168 | 200.60 μs | 143.29 μs | 1.40× | 54.7% |
| 4096 | 8192 | 67.82 μs | 45.39 μs | 1.49× | 58.4% |
| 16384 | 8192 | 221.90 μs | 167.61 μs | **1.32×** | 51.7% |

所有 shape 均明显低于理论的 $2.56\times$。

最高加速出现在

$$
M=256,K=4096,
$$

为 $1.78\times$；进入大规模稳态后，融合收益基本下降到

$$
1.3\sim1.5\times.
$$

---

## 5. 不同 M 范围的性能分析

### 5.1 小 M：固定延迟主导

最明显的证据是：

```text
M=1,  K=4096 : fused = 6.16 us
M=16, K=4096 : fused = 6.16 us
```

M 增加 16 倍，kernel 时间几乎没有变化。

因此这一区间显然没有达到按数据量伸缩的 steady-state throughput，时间主要由固定开销决定，包括：

- kernel launch 和调度开销；
- block reduction；
- warp shuffle；
- block synchronization；
- scale 和 FP4 conversion 的固定执行流水。

Two-step 还需要启动 RMSNorm 和 quant 两个 kernel，而 fused 只有一次 launch，因此融合仍有收益，但此时“减少了多少 B/elem”并不是决定性能的主要因素。

所以 $M=1$ 和 $M=16$ 只有 1.33× 和 1.50×，远低于 2.56× 是正常的。

### 5.2 中等 M：融合收益最大

`M=256,K=4096` 时：

$$
T_{2step}=11.29\ \mu s,
$$

$$
T_{fused}=6.33\ \mu s,
$$

得到最高的

$$
1.78\times.
$$

此时 fused 仍然接近约 6 μs 的低 M execution floor，但 two-step 已经开始随着数据量明显增长。

因此这一区间同时受益于：

1. 减少一次 kernel launch；
2. 消除 bf16 中间值；
3. 减少 global-memory traffic；
4. GPU 已经具有一定 row-level parallelism。

随着 M 继续增加到 1024，fused 时间开始明显增长，说明固定开销逐渐被实际吞吐成本取代。

### 5.3 大 M：进入 steady-state，但不是 DRAM-bound

大 M 下的最终加速为：

```text
16384 × 4096 → 1.45×
16384 × 7168 → 1.40×
16384 × 8192 → 1.32×
```

如果纯字节模型成立，大 M 理应是最接近 2.56× 的区域，但实验结果恰好相反。这说明 kernel 的性能并不满足

$$
T\propto\text{DRAM bytes}.
$$

为分析这一问题，对 `M=16384,K=8192` 使用 Nsight Compute 分别 profile RMS baseline、standalone quant 和 fused kernel。

---

## 6. Nsight Compute 分析

NCU 得到的单 kernel 时间为：

| Kernel | Duration |
|---|---:|
| RMS baseline | 100.32 μs |
| NVFP4 quant | 125.47 μs |
| 两步之和 | 225.79 μs |
| Fused | 168.16 μs |

因此 NCU 下得到

$$
\frac{225.79}{168.16}\approx1.34\times,
$$

与正常 benchmark 中的 1.32× 非常接近。

需要说明的是，NCU 会 replay kernel 以采集不同 counter，因此 profiler 运行时程序自己打印出的端到端计时被严重扰动，不能用于性能比较；这里仅使用 NCU 的单 kernel Duration 和硬件 counter。

### 6.1 Memory 和 Compute 利用率

| Metric | RMS | Quant | Fused |
|---|---:|---:|---:|
| Memory Throughput | 74.64% | 65.22% | **77.69%** |
| DRAM Throughput | **64.54%** | 33.80% | **25.27%** |
| L1/TEX Throughput | 79.23% | 67.93% | **80.64%** |
| L2 Throughput | 47.18% | 33.65% | 31.44% |
| Compute Throughput | 77.34% | 55.74% | 63.23% |

最关键的数据是：

$$
DRAM_{fused}=25.27\%.
$$

融合 kernel 的 DRAM 利用率只有约四分之一，说明它根本不是 HBM bandwidth-bound。

与此同时：

$$
L1/TEX_{fused}=80.64\%,
$$

已经明显高于 DRAM throughput。

NCU 也直接指出 fused 的 memory side 更繁忙，并建议检查 L1 bottleneck。

因此 fusion 发生了明显的 bottleneck migration：

```text
Two-step
    ↓
较高的 DRAM traffic

消除 bf16 中间值
    ↓
DRAM 压力显著下降

Fused
    ↓
瓶颈迁移到 L1/TEX + compute + resource latency
```

这正是 2.56× 理论模型失效的核心原因。

理论模型只计算了 global-memory 字节数，却默认 DRAM bandwidth 是唯一瓶颈。实际 fusion 成功减少 DRAM traffic 后，原来被 DRAM 隐藏的片上访存、conversion、reduction 等成本开始暴露出来，因此无法继续按照字节数线性获得收益。

---

## 7. Occupancy 与寄存器成本

NCU 的 occupancy 数据为：

| Metric | RMS | Quant | Fused |
|---|---:|---:|---:|
| Theoretical Occupancy | 100% | 100% | **75%** |
| Achieved Occupancy | 93.47% | 80.94% | **71.01%** |
| Active Warps / SM | 59.82 | 51.80 | **45.45** |

Fused kernel 的 theoretical occupancy 被明确报告为 register-limited。

编译阶段的 `ptxas -v` 也显示，早期 512-thread 版本：

```text
RMS baseline : 32 registers/thread
Quant        : 31 registers/thread
Fused        : 40 registers/thread
```

并且三者均：

```text
0 bytes spill stores
0 bytes spill loads
```

因此问题不是 register spilling，而是：

> Fusion 增加了每线程的 live register set，使得一个 SM 能同时驻留的 warp 数下降。

融合 kernel 中需要同时维护 RMS reduction 状态、`rnorm`、16-element group 的 FP32 中间值、`amax`、scale、FP4 conversion 等状态，因此 resource footprint 必然高于独立的 RMS 或 quant kernel。

这导致 fused 的 active warps 从 RMS 的约 60 warp/SM、quant 的约 52 warp/SM 降至约 45 warp/SM，进一步削弱了隐藏 memory 和 instruction latency 的能力。

---

## 8. 为什么没有达到 2.56×

以 `16384×8192` 为例：

$$
T_{2step}=221.90\ \mu s,
$$

如果理想的 2.56× 字节模型完全成立，则预测：

$$
T_{\mathrm{ideal}}
=
\frac{221.90}{2.56}
\approx86.7\ \mu s.
$$

而实际：

$$
T_{\mathrm{fused}}=167.61\ \mu s.
$$

也就是说，理论模型认为 fusion 应该节省约

$$
221.90-86.7=135.2\ \mu s,
$$

而实际上只节省：

$$
221.90-167.61=54.29\ \mu s.
$$

理论可节省时间只有大约 40% 被真正兑现。

剩余收益主要消耗在以下几个方面：

1. fused 已经不是 DRAM-bound，DRAM throughput 只有 25.27%；
2. L1/TEX throughput 达到 80.64%，瓶颈转移到片上 memory pipeline；
3. RMS reduction 和 block synchronization 并不会因 fusion 消失；
4. `amax`、E4M3 scale、FP8/FP4 conversion 等计算成本仍然存在；
5. fusion 增大寄存器 footprint，theoretical occupancy 从 100% 降为 75%；
6. baseline 在公平调参后本身已经非常快，因此简单比较“2 个 kernel 和 1 个 kernel”会高估融合收益。

因此，2.56× 应理解为一个仅由数据字节数得到的 **memory-only upper bound**，而不是实际 kernel 的性能预测。

---

## 9. 相对 5.3(c) Ceiling Probe 的性能

5.3(c) ceiling probe 保持与 NVFP4 quant 相同的内存形状，但去除了 RMS reduction、scale 计算和 FP4 conversion 等数学操作。其理论访存量为

$$
2+\frac12+\frac1{16}
=
2.5625\ \mathrm{B/elem},
$$

与本题 fused kernel 的字节账基本一致。因此 probe 可以近似表示：在相同输入输出 memory shape 下，如果只考虑数据搬运，能够达到的性能上限。

实测结果如下：

| M | K | Probe 时间 | Probe 带宽 | Fused 时间 | Fused / Probe |
|---:|---:|---:|---:|---:|---:|
| 4096 | 7168 | 22.59 μs | 3330 GB/s | 41.04 μs | **55.0%** |
| 16384 | 4096 | 45.04 μs | 3818 GB/s | 84.05 μs | **53.6%** |
| 16384 | 8192 | 84.30 μs | 4080 GB/s | 167.61 μs | **50.3%** |

由于两者处理相同 shape，并具有相同的理论字节数，因此性能比例可以直接由运行时间得到：

$$
R_{\mathrm{ceiling}}
=
\frac{t_{\mathrm{probe}}}
     {t_{\mathrm{fused}}}.
$$

例如 `M=16384,K=8192`：

$$
R_{\mathrm{ceiling}}
=
\frac{84.30}{167.61}
\approx50.3\%.
$$

因此，在大规模 shape 下，fused kernel 实际只能达到约 **50%–55% 的纯 memory-shape ceiling**。

这一结果进一步证明，fused kernel 的运行时间并不是由理论字节数单独决定的。若其性能完全由内存搬运控制，那么 fused 应当接近 5.3(c) probe；但实际耗时约为 probe 的两倍。

对应的 fused 有效带宽约为：

$$
3330\times55.0\%\approx1.83\ \mathrm{TB/s},
$$

$$
3818\times53.6\%\approx2.05\ \mathrm{TB/s},
$$

$$
4080\times50.3\%\approx2.05\ \mathrm{TB/s}.
$$

尤其是 `16384×8192`，probe 已达到约 4.08 TB/s，而 fused 只有约 2.05 TB/s。两者具有相同的内存字节形状，因此这约一半的差距不能归因于额外 HBM 流量，而来自 fused kernel 本身必须执行的工作，包括：

- RMS 的平方和归约与 block synchronization；
- `rnorm` 计算；
- RMSNorm 乘法；
- 每 16 元素的 `amax` 计算；
- E4M3 scale conversion；
- E2M1 FP4 conversion；
- 更大的寄存器 live set 和由此造成的 occupancy 降低。

这一结果与 Nsight Compute 的观察一致。对于 `16384×8192`，fused 的 DRAM throughput 只有 25.27%，而 L1/TEX throughput 达到 80.64%，说明它并没有接近 HBM bandwidth ceiling。与此同时 fused 的 achieved occupancy 只有 71.01%，低于独立 RMS 和 quant kernel。

因此可以把 5.3(c) probe 和 NCU 两组证据结合起来得到更强的结论：

> fused kernel 虽然已经把理论 HBM traffic 降到了与 ceiling probe 相同的水平，但实际只能达到 probe 约一半的吞吐。这说明融合后性能瓶颈已经从“需要搬多少字节”转移到了 reduction、片上 memory pipeline、quant conversion 和寄存器资源等 kernel 内部开销。

---

## 10. 结论

本实验实现了正确的 fused RMSNorm + NVFP4 quant kernel，并在 B300 上对 fused 与 two-step baseline 分别进行了独立调优。

实验首先说明了公平 baseline 的重要性。未调优时，fusion 一度可以表现出接近 2× 的加速，但在对 RMSNorm、standalone quant、grid 和 block size 分别优化后，大规模 shape 的真实收益下降到约 1.3–1.5×。因此，如果 baseline 本身处于不利配置，会明显高估 fusion 的收益。

不同 M 区间的限制因素不同：小 M 主要受 kernel launch、调度和 reduction 等固定延迟影响；中等 M 下固定开销开始被摊薄，同时 fusion 消除了 bf16 中间值，因此出现最高约 1.78× 的收益；大 M 则进入稳定吞吐区域，但 Nsight Compute 表明 fused 的 DRAM throughput 仅为 25.27%，而 L1/TEX throughput 已达到 80.64%，同时 theoretical occupancy 因寄存器占用下降至 75%。

因此，融合确实成功减少了显存流量，但同时使瓶颈从 HBM traffic 转移到了片上 memory pipeline、计算、同步和寄存器资源。理论上的 2.56× 只描述了字节数减少带来的上限，而没有包含这种 bottleneck migration。

这也解释了相关 fused kernel 在上游工程中可能出现的现象：kernel 数量从两个减少为一个、理论访存量也显著下降，并不意味着端到端运行时间会按相同比例下降。只有当被消除的 HBM traffic 本身确实位于主要性能瓶颈上时，fusion 的理论访存收益才能充分转化为实际加速。


Optional：根据分析得到的主要瓶颈进行一次针对性优化，重新测试并更新表格。

:::


### 5.5 {.prob type=CONCEPT}

比较 W4A16 + Marlin kernel（权重 int4、计算 fp16）与 5.3 中的
NVFP4 GEMM。

根据课件 S109 的分类回答：

(a) 两者分别属于存储量化还是计算量化？

(b) 两种方法分别节省哪些资源：显存容量、显存带宽还是计算吞吐？

(c) 在 4.5 中的小 batch decode 场景下，哪一类量化的收益更加直接？

每问用两到三句话回答。


# TileLang 对照

前面的模块已经手动处理过 Tensor Core 指令、数据布局和数据搬运。
本模块通过 TileLang 的 lowering 结果观察其中哪些工作可以交给 DSL 完成。

::: reading
课件 S112；assignment01 Module 7（7.5 的表、7.6 的
`kernels/tilelang_matmul.py`）；TileLang 文档中的 lowering/debug 部分。
:::


### 6.1 {.prob type=EXPERIMENT}

使用 assignment01 的 `kernels/tilelang_matmul.py`，或任意使用
`T.gemm` 的 kernel，分别以 `sm_90a` 和 `sm_100a` 为 target 编译。
本题只要求编译，不需要实际使用 H 卡。

保存生成的 CUDA 源码与 lowering 输出，并填写：

| | sm_90a | sm_100a |
|---|---|---|
| 选中的 Tensor Core 指令 | | |
| descriptor 在哪里、由谁生成 | | |
| smem swizzle 布局在哪一步确定 | | |
| 数据由谁搬入 smem | | |

与 M2--M4 中的手写实现进行对照，并回答：

(a) 哪些硬件相关的决策已经由 DSL 自动完成？

(b) 哪些参数仍然需要程序员决定，例如 tile 尺寸和 stages？

最后在 assignment01 7.5 的“谁负责”表中补充一行：

`Tensor Core 指令选择与供数布局`

<!-- 编者注：TileLang 对 sm_100 codegen 的支持范围需在发布前根据
README 中固定的版本重新核实。 -->


# 团队选做（推荐） {-}

2--4 人一组，从 `team/` 中的两个题目任选一个，也可以全部完成：

- C1：FlashKDA 官方 kernel 当前使用 SM80 MMA，分析迁移到 SM100 是否值得；
- C2：MiniMax M3 MSA 的小 batch decode 当前使用 Triton，分析是否值得实现专用 kernel。

每道题分为三个阶段：

`复现与测量 → 分析（结论 + 证据）→ 挑战`

挑战阶段不要求一定得到正向加速。如果实验结果表明继续优化收益有限，
只要能够用数据说明为什么不值得继续做，同样视为有效结论。

答辩时间为 10 分钟，另有 5 分钟提问；提问优先由选择另一道团队题的
小组提出。具体要求见 `team/README.md`。

团队题不限制 AI 工具的使用。


# 提交 {-}

- **代码**：提交所有动手题的实现与判测输出；FROM-SCRATCH 题同时保留判测脚本的 PASS 记录。
- **报告**：包含纸面题解答、实验表格与性能归因，以及 DEBUG 题的现象记录和修改说明。所有实验数据注明使用的 GPU。
- **团队题**：提交代码、报告并完成答辩，具体要求见 `team/README.md`。
