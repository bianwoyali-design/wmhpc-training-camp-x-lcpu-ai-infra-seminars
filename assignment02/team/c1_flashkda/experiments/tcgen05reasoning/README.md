# CHUNK=16：SM80 与 tcgen05 指令替换

讨论点 2：固定 CHUNK=16、D=128，比较同一局部矩阵乘的 SM80 与 tcgen05 实现。
阅读顺序：[分析](ANALYSIS.md) → [正式结果](results/SUMMARY.md) → `csrc/compare.cu`。

## 复现

需要 B300、CUDA 13（nvcc、nvdisasm、compute-sanitizer）与 CUDA PyTorch，无额外 Python 包。
在仓库根目录执行：

```bash
srun -p gpu -G 1 --time=00:20:00 \
  assignment02/.venv/bin/python \
  assignment02/team/c1_flashkda/experiments/tcgen05reasoning/run.py reproduce
```

默认输出到 `runs/<时间戳>/`，也可用 `--out` 指定不存在的新目录。
流程包含编译、反汇编、memcheck、racecheck、数值对拍、计时和汇总。
在本目录、激活虚拟环境后：

```bash
python run.py build
python run.py inspect
srun -p gpu -G 1 python run.py check
srun -p gpu -G 1 python run.py reproduce --quick
python run.py report --out results
```

使用显式 `-gencode arch=compute_103a,code=sm_103a`，避免 CUDA 13.0 的 `-arch=sm_103a`
简写额外生成不支持架构专属指令的通用 PTX。编译缓存位于 `build/`。

## 测什么

输入 A[M,K]、B[K,N] 为相同 BF16 张量，所有实现输出同样的 FP32 C[M,N]。
装载、补零、shared-memory 布局转换、TMEM 分配/释放、同步和输出回写均计入 kernel 时间。

| shape | M,N,K | 来源 |
|---|---|---|
| gram | 16,16,128 | K1 的 Gram / Mqk |
| projection | 16,128,128 | K2 的 K/Q × state |
| local_mix | 16,128,16 | K2 的 INV/Mqk × U |
| state_update | 128,128,16 | K2 的状态更新 |
| square | 16,16,16 | 局部方阵乘的形状探针；不等同于 FP16 Neumann 求逆 |

| 实现 | 用途 |
|---|---|
| sm80_w1/w4/w8 | WMMA 16×16×16 → HMMA；扫描 CTA 的 warp 数 |
| sm80_pad_w4 | 把 M<64 补到 64，观察增加无效算术量的影响 |
| sm80_transpose_w4 | 计算 Cᵀ=BᵀAᵀ，作为布局改变的 SM80 控制组 |
| tc_direct | 原方向，M<64 补到 64，cta_group::1 |
| tc_transpose | 计算 Cᵀ=BᵀAᵀ，转置写回原接口；仍固定 CHUNK=16 |

所有路线都用独立矩阵和 FP32 累加，不修改 FlashKDA 内核。
SM80 最佳候选在原方向 1/4/8 warp 与转置 4 warp 中选择，补零组仅作控制。

## 检查与计时

- 正确性：三个种子，每组八个矩阵；随机 BF16、小整数和零输入；repeats=1/32。
  参考为量化后输入的 FP64 GEMM。随机相对 Frobenius 误差 ≤2e-5，整数/零结果必须精确相等。
- memcheck/racecheck 使用一个种子、三个矩阵，覆盖全部 shape/method/repeats。
- batch=1/12/96/6144/49152；后两者是 8192/16×12/96 个独立 chunk-head。
  **K2 存在状态依赖，不能把这两个大 batch 当作 K2 可用并行度。**
- 默认每张 CUDA Graph 30 次调用，五轮随机方法顺序，每轮五次 CUDA Event 测量。
  同一 shape/batch 的所有实现共享输入输出缓冲区；五次预热，测量前额外预热一次 Graph。
- batch≤96 另测 repeats=32：同一 CTA 装载一次，重复累加 D+=AB，最后回写一次。
  这是固定输入累加对照，不包含 KDA 的状态更新或中间 BF16 舍入，也不代表完整融合实现。
- 记录每轮 median/min/max，汇总为轮次中位数的中位数；每 200 ms 采样频率、功耗、温度。
- `--quick` 只测 batch=1/12/96，两个轮次、每轮两次测量、10 次调用/Graph；不用于正式结论。

`resources.csv` 的 CTA 驻留上限来自 CUDA occupancy API；该数字不能单独刻画动态 TMEM
分配限制。`instructions.json` 是静态指令证据，不是动态执行计数。

## 文件

```text
run.py             构建、正确性、计时与汇总
csrc/compare.cu    两类指令和补零/转置控制组
ANALYSIS.md        纸面推算、实测与结论边界
results/           正式 CSV、配置、指令证据和 sanitizer 日志
build/             构建缓存，不提交
runs/              新运行结果，不提交
```

正式结果只有 `metadata.json` 中 `status=complete` 且两项 sanitizer 为 passed 才用于分析。
性能是局部算子在热缓存下的测量，不能直接外推为 FlashKDA 端到端加速。
