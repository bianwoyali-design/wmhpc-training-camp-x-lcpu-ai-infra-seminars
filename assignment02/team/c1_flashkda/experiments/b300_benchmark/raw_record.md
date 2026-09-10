# B300 benchmark

## `T=8192`, `H=96`, `D=128`

```
shape=[8192,96,128] warmup=30 iters=200 repeats=5
  flash_kda (bf16 state) : mean=1.7559 ms, min=1.7473 ms, max=1.9915 ms
  flash_kda (no state)   : mean=1.7639 ms, min=1.7504 ms, max=2.0036 ms
  flash_kda (fp32 state) : mean=1.7142 ms, min=1.7021 ms, max=1.9462 ms
  chunk_kda : mean=3.6052 ms, min=3.5876 ms, max=4.2108 ms
  chunk_gated_delta_rule : mean=1.9777 ms, min=1.9643 ms, max=2.2695 ms
varlen shape=[8192,96,128] seq_lens=[1300, 547, 2048, 963, 271, 3063] warmup=30 iters=200 repeats=5
  flash_kda (bf16 state) : mean=1.4761 ms, min=1.4075 ms, max=1.7383 ms
  flash_kda (no state)   : mean=1.4737 ms, min=1.3909 ms, max=1.6980 ms
  flash_kda (fp32 state) : mean=1.4988 ms, min=1.4205 ms, max=1.7152 ms
  chunk_kda : mean=3.6986 ms, min=3.6796 ms, max=4.2064 ms
  chunk_gated_delta_rule : mean=1.9740 ms, min=1.9604 ms, max=2.2206 ms
varlen shape=[8192,96,128] seq_lens=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024] warmup=30 iters=200 repeats=5
  flash_kda (bf16 state) : mean=1.1864 ms, min=1.1811 ms, max=1.2981 ms
  flash_kda (no state)   : mean=1.1836 ms, min=1.1779 ms, max=1.2564 ms
  flash_kda (fp32 state) : mean=1.2134 ms, min=1.2067 ms, max=1.2882 ms
  chunk_kda : mean=3.5973 ms, min=3.5796 ms, max=3.9536 ms
  chunk_gated_delta_rule : mean=1.8784 ms, min=1.8662 ms, max=2.0814 ms
```

## `T=8192`, `H=64`, `D=128`

```
shape=[8192,64,128] warmup=30 iters=200 repeats=5
  flash_kda (bf16 state) : mean=1.6076 ms, min=1.6010 ms, max=1.8299 ms
  flash_kda (no state)   : mean=1.6128 ms, min=1.6010 ms, max=1.6929 ms
  flash_kda (fp32 state) : mean=1.5612 ms, min=1.5506 ms, max=1.6076 ms
  chunk_kda : mean=2.4731 ms, min=2.4581 ms, max=2.9227 ms
  chunk_gated_delta_rule : mean=1.3339 ms, min=1.3222 ms, max=1.4991 ms
varlen shape=[8192,64,128] seq_lens=[1300, 547, 2048, 963, 271, 3063] warmup=30 iters=200 repeats=5
  flash_kda (bf16 state) : mean=1.1145 ms, min=1.1073 ms, max=1.1667 ms
  flash_kda (no state)   : mean=1.1173 ms, min=1.1094 ms, max=1.2669 ms
  flash_kda (fp32 state) : mean=1.1280 ms, min=1.1197 ms, max=1.3782 ms
  chunk_kda : mean=2.5785 ms, min=2.5563 ms, max=2.7515 ms
  chunk_gated_delta_rule : mean=1.4305 ms, min=1.4197 ms, max=1.6199 ms
varlen shape=[8192,64,128] seq_lens=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024] warmup=30 iters=200 repeats=5
  flash_kda (bf16 state) : mean=0.8033 ms, min=0.7990 ms, max=0.8271 ms
  flash_kda (no state)   : mean=0.8008 ms, min=0.7964 ms, max=0.8304 ms
  flash_kda (fp32 state) : mean=0.8221 ms, min=0.8166 ms, max=0.8517 ms
  chunk_kda : mean=2.3831 ms, min=2.3629 ms, max=2.5311 ms
  chunk_gated_delta_rule : mean=1.2669 ms, min=1.2568 ms, max=1.3955 ms
```