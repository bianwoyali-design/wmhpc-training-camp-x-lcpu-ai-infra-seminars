# 讨论点 4：FlashKDA 的瓶颈在哪里

[分析](ANALYSIS.md) · [NCU 指标表](results/SUMMARY.md)

对官方 K1（prepare）和 K2（recurrence）分别采集 NCU 指标，结合未插桩计时、
每卡 head 数、独立序列数和缓存对照，区分 Tensor 吞吐、HBM 带宽与并行度/延迟限制。
使用 BF16 初始及最终状态，CHUNK=16、D=128；不修改官方 CUDA 内核。

## 复现

需要已安装的 FlashKDA、CUDA PyTorch、einops、CUDA 13、Nsight Compute 和 B300，
以及 GPU 性能计数器权限。从仓库根目录执行：

```bash
bench=assignment02/team/c1_flashkda/experiments/bottleneckreasoning
out="$bench/runs/$(date -u +%Y%m%dT%H%M%S)"
srun -p gpu -G 1 --time=00:40:00 bash -c '
  assignment02/.venv/bin/python "$1/run.py" reproduce --out "$2" &&
  assignment02/.venv/bin/python "$1/memory.py" --out "$2/memory"
' bash "$bench" "$out"
```

结果默认写入 `runs/<时间戳>/`；`--out` 可指定尚不存在的新目录。快速检查只测一组冷缓存：

```bash
srun -p gpu -G 1 --time=00:10:00 \
  assignment02/.venv/bin/python \
  assignment02/team/c1_flashkda/experiments/bottleneckreasoning/run.py reproduce --quick --case h12_n1

python3 assignment02/team/c1_flashkda/experiments/bottleneckreasoning/run.py report \
  --out assignment02/team/c1_flashkda/experiments/bottleneckreasoning/results
```

`report` 只解析 CSV，不需要 GPU。用 `ncu-ui results/<case>__<cold|warm>.ncu-rep` 查看原始报告；
`metadata.json` 保存了每次 NCU 调用的完整命令。

## 实验矩阵与口径

总 token 数固定为 8192。等长输入为 `[N,8192/N,H,128]`，变长输入为 `[1,8192,H,128]`。

| case | 每卡 H | 独立序列 N | 长度 | K2 CTA 数 |
|---|---:|---:|---|---:|
| h12_n1 | 12 | 1 | 8192 | 12 |
| h12_n8 | 12 | 8 | 8×1024 | 96 |
| h96_n1 | 96 | 1 | 8192 | 96 |
| h96_n8 | 96 | 8 | 8×1024 | 768 |
| h96_skew8 | 96 | 8 | 4608 + 7×512 | 768 |

这些是独立序列工作负载对照；把一条序列切成八条会改变语义，不是可直接应用的优化。

- **cold**：kernel replay，`--cache-control all`，每次采集清缓存。
- **warm**：application replay，`--cache-control none`，每次应用重放先执行十次完整 forward，
  保存输出/状态快照，再采集一次 K1/K2。保留这段应用形成的缓存状态，不保证整个工作集留在缓存。
- 两者都使用 `--clock-control none`。记录 NCU 内的平均频率，以及按 UUID 每 200 ms 的
  `nvidia-smi` 频率/功耗/温度；不把频率不同的结果直接归因于内核。
- NCU 仅选择 profiler 区间内的 K1/K2，各采集一次，多个 replay pass 收集全部计数器。
  beta 转置、varlen prefix 不在 NCU 分阶段表中。
- 正常计时复用[讨论点 3](../parallelismreasoning/README.md)的 CUDA Graph 节点隔离工具：
  每图十次 forward，五轮随机 full/K1/K2 顺序，每轮五次 CUDA Event 测量。
  保存每轮中位数、最小值和最大值。full 包含辅助内核，但不包含 Python 分配/dispatch。
- 正常计时和 NCU 使用相同形状、dtype、归一化及输入分布，随机种子不同；
  NCU 各 replay 使用固定种子 1234。每次 NCU target 检查输出/状态有限，且与预热结果逐位一致；
  Graph 工具另行检查隔离阶段与完整调用逐位一致。这里验证采集没有破坏输出，
  不替代讨论点 5 的参考精度验证。

## 文件

| 文件 | 用途 |
|---|---|
| `run.py` | 复现、单次采集 target、CSV 汇总；依赖讨论点 3 的 Graph helper |
| `memory.py` | H=96、N=1/8 的冷缓存 L1TEX/L2/SM breakdown，直接复用同一个 target |
| `ANALYSIS.md` | 纸面成本、实测证据、指标解释及结论边界 |
| `results/*.ncu-rep` | 可用 NCU GUI 打开的原始报告 |
| `results/*.raw.csv` | NCU 原始导出，含 launch/occupancy 与单位 |
| `results/metrics.csv` | 规范化长表，保留全部导出指标 |
| `results/summary.csv`、`SUMMARY.md` | 分阶段主要指标与派生吞吐 |
| `results/normal_timing.csv` | 未插桩的热缓存 Graph 计时，单位 µs |
| `results/*.check.json`、`*.log` | 输出检查与采集日志 |
| `results/metadata.json`、`clocks.csv` | 环境、哈希、命令、运行状态及频率记录 |
| `results/memory/` | 细分采集的报告、CSV、检查、命令与代码哈希；频率见其 NCU 指标 |

`build/`、`runs/` 为忽略的本地缓存和新实验目录。正式证据集中在 `results/`。
