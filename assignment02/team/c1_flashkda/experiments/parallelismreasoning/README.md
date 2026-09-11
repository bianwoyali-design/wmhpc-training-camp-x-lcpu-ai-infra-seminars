# 讨论点 3：递推的并行度从哪里来

[分析与方案反例](ANALYSIS.md) · [官方 K1/K2 测量](results/SUMMARY.md)

本实验包含官方内核的并行度扫描、候选拆分的代数对照和理想调度模型。
未实现多 head/CTA、动态 persistent 或 2-CTA 的完整 FlashKDA 替代内核；不把模型预测当作 GPU 加速。

## 复现

需要已安装的 FlashKDA、CUDA PyTorch、einops、CUDA 13 工具链和 B300。
从仓库根目录执行：

```bash
srun -p gpu -G 1 --time=00:20:00 \
  assignment02/.venv/bin/python \
  assignment02/team/c1_flashkda/experiments/parallelismreasoning/run.py reproduce
```

结果默认写入 `runs/<时间戳>/`，`--out` 可指定不存在的新目录。在本目录激活环境后：

```bash
python run.py check                     # CPU：代数正确性和反例
python run.py build                     # 只构建 CUDA Graph 检查工具
srun -p gpu -G 1 python run.py reproduce --quick
python run.py report --out results
```

## 测量口径

- 总 token 数固定 8192，H=12/24/48/96，D=128，独立序列数 N=1/2/4/8/16。
  等长模式使用 `[N,8192/N,H,128]`；这些是不同的独立序列工作负载。
- 变长对照：`[4608]+[512]*7`、官方六段长度、`[1024]*8`。
  加入等长 varlen，区分序列不均衡与 varlen 代码路径本身。
- H=12 另测单序列 T=128/512/2048，观察递推长度。
- q/k 归一化，BF16 q/k/v/g/beta，随机小 BF16 初始状态与 BF16 最终状态；gate 激活与 beta sigmoid 仍由官方完成。
- 预分配 workspace，捕获官方扩展调用到 CUDA Graph。每张 Graph 默认 10 次 forward，
  五轮随机 `full/k1/k2` 顺序，每轮五次 CUDA Event 测量；各轮中位数再取中位数。
- `full` 包含 beta 转置、varlen prefix 和 K1/K2。隔离 K1/K2 时通过 CUDA Graph API
  禁用其他 kernel 节点，之前已运行完整 graph 填好输入/工作区；禁用的节点等价于空节点。
- 每个 case 验证隔离 K2，以及 K1 后再 K2 的输出和最终状态，都与 full **逐位一致**。
  计时不包含 Python 调用和分配；是热缓存 GPU 图重放，不等同于官方脚本的 eager benchmark。
- 记录节点 grid/block、寄存器、共享内存和 occupancy API 上限；后者不是动态 achieved occupancy。
- 每 200 ms 采样 GPU UUID 对应的频率、功耗与温度。`--quick` 缩小 case 与采样数，仅作流程检查。

## 三类证据

| 文件 | 内容 |
|---|---|
| `results/timing.csv` | 官方 full/K1/K2 每轮耗时，单位 µs |
| `results/kernels.csv` | 捕获的真实 kernel 名称、grid、资源与驻留上限 |
| `results/checks.csv` | 每个 case 的隔离重放输出/状态检查 |
| `results/algebra.csv` | 对提供的 `naive_recurrent_kda` 检查 value/head/time/key 拆分，以及 affine scan |
| `results/scheduling_model.csv` | FIFO、最长任务优先、静态轮转的理想模型，单位 chunk，不是实测时间 |
| `results/metadata.json` | 设备、版本、已加载扩展二进制哈希、代码哈希及运行状态 |
| `results/clocks.csv` | 测量期间的时钟记录 |

CPU 代数检查使用三个种子，B=2、T=32、H=4、K=V=128，非零初始状态；
value 拆成 2/4/8 份，head 分组 1/2。affine scan 为便于检查使用 K=V=8。
正确方案整体相对误差须小于 2e-5；错误拆分必须产生大于 0.01 的误差，防止退化输入掩盖错误。
这些验证的是数学可分性，不是尚未实现的新 CUDA kernel 的精度证明。

理想调度模型把一条 sequence/head 视为不可抢占任务，成本为 ceil(length/16)。
148/296 个 slot 分别代表一/两条 CTA 驻留的抽象，并不意味着两条 CTA 能将 SM 算力翻倍。
CUDA 不保证按模型中的 FIFO 顺序调度；此模型用于构造反例和下界，不预测官方实测时间。

```text
run.py          复现入口、官方内核捕获、计时与调度模型
proofs.py       对朴素参考的代数检查
csrc/graph.cpp  通过 CUDA Driver API 读取节点信息，通过 Runtime API 启停节点
ANALYSIS.md     结论、方案成立条件、反例与后续验证要求
results/        正式证据
build/、runs/   构建缓存及新结果，不提交
```
