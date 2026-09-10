# 为什么选择 CHUNK=16

量化 CHUNK=16/32/64 的数值范围、Neumann 求逆成本和 MMA 形状。
先读 [分析与结论](ANALYSIS.md)，再按需查看 [测量汇总](results/benchmark/SUMMARY.md) 和 CSV。

## 复现

需要 B300、CUDA 13（nvcc、nvdisasm、compute-sanitizer）、CUDA PyTorch，
以及 `../../FlashKDA/cutlass/include`（CUTLASS commit `5c149f52a436782210263fb2f19b354443a61c6a`）。
无需 matplotlib 或额外安装实验包。在仓库根目录执行：

```bash
srun -p gpu -G 1 --time=00:20:00 \
  assignment02/.venv/bin/python \
  assignment02/team/c1_flashkda/experiments/chunk16reasoning/run.py reproduce
```

流程：构建 → FTZ 指令对照 → racecheck → 数值检查 → 计时 → 汇总。
结果写入 `runs/<时间戳>/`，终端会打印路径；`--out` 可指定不存在的新目录。
脚本不依赖当前工作目录。以下命令在本目录、激活虚拟环境后运行：

```bash
python run.py build                             # 提前编译，无需 GPU
python run.py inspect                           # PTX/SASS 静态指令检查
srun -p gpu -G 1 python run.py validate           # 仅 FTZ/CPU 模型对照
srun -p gpu -G 1 python run.py check              # 仅三组实验的数值检查
srun -p gpu -G 1 python run.py reproduce --quick  # 冒烟检查，不用于性能结论
srun -p gpu -G 1 python run.py reproduce --suite inverse --warps 1,4,8
python run.py report results/benchmark          # 从 CSV 重新汇总
python3 -m unittest discover -s tests -v
```

`--suite` 支持 `range/inverse/mma/all`，默认 all；`--warps` 支持 1/4/8 的子集。
`bench` 执行数值检查与计时，不运行 FTZ 对照或 racecheck。
`reproduce --skip-racecheck` 可显式跳过 sanitizer，配置会记录这一点。

## 实验设计

| 实验 | 对照与输入 | 检查 |
|---|---|---|
| FTZ | FP64 参考、CPU FTZ/BF16 模型与实际 CUDA 指令；五种常数 gate、五个种子的均匀/混合输入 | 180 组 GPU/模型比较；零值、Inf、NaN、有限值差异；最坏 gate 第 18 token 失效 |
| range | 原始因子、整块居中、tile16 重缩放、直接指数；常数 0/-0.1/-1/-3/-5、均匀与混合 gate | FP64 因果衰减矩阵参考；异常数和整体相对误差；零 gate 精确为 1 |
| inverse | 官方 C=16 包装与统一 C=16/32/64、1/4/8 warp；structured、零矩阵、高斯 stress 严格下三角 L | FP64 triangular solve；残差、解误差、warp 逐位一致 |
| mma | BF16 Gram、状态投影、局部混合、状态更新；三种尺寸、三种 warp | FP64 GEMM 参考、warp 逐位一致、padding、生成的 PTX/SASS |

FTZ 的 CPU 模型模拟输出冲零，不模拟近似指数指令的全部细节；两者匹配仅对已测输入成立。
range/inverse/mma 默认三个种子、每组八个矩阵，常数输入不重复种子。
range 使用同一长度 64 输入的前缀；只统计因果位置，非有限误差留空。
tile16/direct 相对误差阈值 0.02；inverse structured 残差阈值 0.02；MMA 相对误差阈值 1e-5。
这些阈值用于发现实验故障，不是模型精度验收标准。

统一求逆使用 FP16 输入、FP32 WMMA 累加，每次完整 GEMM 与加法后舍入 FP16，最终输出 BF16。
FP64 参考使用量化后的 L。官方包装保留原 FP16 累加与融合组织，仅作 C=16 参照。
racecheck 覆盖新实现 unified/decay/shape_mma；官方包装参与数值检查。

## 计时口径

- 工作量：单矩阵、固定 8192 tokens、固定 8192×96 tokens；batch=总 token/C。
- 输入输出预分配并反复复用；10 次预热，100 次调用/CUDA Graph。
- 五轮随机配置顺序，每轮五次 CUDA Event 测量。汇总取轮次中位数的中位数，范围不是置信区间。
- range 计时使用温和 gate，使原始基线有效；每个 suite 每 200 ms 采样 GPU 时钟、功耗和温度。
- `--quick` 缩为一个种子、四个矩阵、10 次调用/Graph、两轮各两次测量，只测 single/8192。

局部算子测量含装载、输出和同步，使用热缓存。MMA 不模拟跨 chunk 状态依赖，
range 重缩放只修复衰减矩阵；不能把局部时间相加当作完整 KDA 耗时。

## 文件

```text
run.py          唯一命令入口，构建与流程
experiment.py   三组微基准的输入、检查和计时
numerics.py     FTZ 的 CPU 参考与 GPU 结果对照
report.py       CSV → SUMMARY.md
csrc/           decay、unified、official、mma、ftz CUDA 实现
results/        支撑分析的完整测量 benchmark/ 与指令对照 ftz/
runs/           新运行结果（不提交）
build/          构建缓存（不提交）
tests/          汇总器与 CLI 检查
```

结果包含 `<suite>_accuracy.csv`、`<suite>_timing.csv`（µs）、`<suite>_clocks.csv`、
`inverse_resources.csv`、`instructions.json`、编译/racecheck 日志和 `metadata.json`。
配置保留设备、版本、源码哈希、采样设置和完成状态。引用性能前检查 `status=complete`、
`racecheck=passed` 和频率；优先比较同次运行的比值。
