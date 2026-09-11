# FlashKDA 实验与讨论

每个目录的 README 给出复现命令，ANALYSIS 给出结论与边界，results 保留正式数据。

| 讨论点 | 实验与分析 |
|---|---|
| 1：CHUNK=16 的范围、求逆与 MMA 形状 | [chunk16reasoning](chunk16reasoning/README.md) |
| 2：tcgen05 形状与局部收益 | [tcgen05reasoning](tcgen05reasoning/README.md) |
| 3：状态依赖与并行度候选 | [parallelismreasoning](parallelismreasoning/README.md) |
| 4：计算、HBM 与片上访存瓶颈 | [bottleneckreasoning](bottleneckreasoning/README.md) |
| 5：BF16 状态精度与压力反例 | [stateprecisionreasoning](stateprecisionreasoning/README.md) |
| 6 + 任务 3：完整 value-split K2 挑战与 v2 决策 | [v2decision](v2decision/README.md) |

讨论点 6 的完整 forward 实验在 H12 单长序列上取得 1.208× 加速，但高并行度组回退；
据此建议保留通用路径，研发按形状分派的优化后端，而非全量替换。
