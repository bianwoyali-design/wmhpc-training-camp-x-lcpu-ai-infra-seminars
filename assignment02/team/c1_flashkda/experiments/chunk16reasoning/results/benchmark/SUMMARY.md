# CHUNK 实验结果

设备：NVIDIA B300 SXM6 AC；状态：complete；Slurm：25034。

耗时是各轮中位数的中位数；round range 是轮次中位数范围，不是置信区间。
每个候选只与同一次运行、相同 workload/stage 的结果比较；不混合 eager 与 graph。

## range

| workload | chunk | method | mode | µs/call | round range µs | rounds |
|---|---|---|---|---|---|---|
| single | 16 | center | graph | 1.694 | 1.689–1.696 | 5 |
| single | 16 | direct | graph | 1.736 | 1.732–1.739 | 5 |
| single | 16 | raw | graph | 1.693 | 1.688–1.695 | 5 |
| single | 16 | tile16 | graph | 1.715 | 1.708–1.716 | 5 |
| single | 32 | center | graph | 1.859 | 1.859–1.870 | 5 |
| single | 32 | direct | graph | 1.980 | 1.972–1.983 | 5 |
| single | 32 | raw | graph | 1.857 | 1.856–1.858 | 5 |
| single | 32 | tile16 | graph | 2.429 | 2.412–2.430 | 5 |
| single | 64 | center | graph | 3.740 | 3.735–3.745 | 5 |
| single | 64 | direct | graph | 4.355 | 4.353–4.357 | 5 |
| single | 64 | raw | graph | 3.700 | 3.694–3.703 | 5 |
| single | 64 | tile16 | graph | 5.316 | 5.312–5.320 | 5 |
| tokens_786432 | 16 | center | graph | 26.794 | 26.793–26.802 | 5 |
| tokens_786432 | 16 | direct | graph | 26.814 | 26.812–26.817 | 5 |
| tokens_786432 | 16 | raw | graph | 26.777 | 26.773–26.784 | 5 |
| tokens_786432 | 16 | tile16 | graph | 26.814 | 26.812–26.821 | 5 |
| tokens_786432 | 32 | center | graph | 22.907 | 22.907–22.924 | 5 |
| tokens_786432 | 32 | direct | graph | 23.308 | 23.295–23.325 | 5 |
| tokens_786432 | 32 | raw | graph | 22.886 | 22.884–22.903 | 5 |
| tokens_786432 | 32 | tile16 | graph | 23.298 | 23.297–23.316 | 5 |
| tokens_786432 | 64 | center | graph | 42.976 | 42.964–42.989 | 5 |
| tokens_786432 | 64 | direct | graph | 43.594 | 43.574–43.610 | 5 |
| tokens_786432 | 64 | raw | graph | 42.959 | 42.945–42.967 | 5 |
| tokens_786432 | 64 | tile16 | graph | 44.749 | 44.728–44.755 | 5 |
| tokens_8192 | 16 | center | graph | 1.859 | 1.852–1.860 | 5 |
| tokens_8192 | 16 | direct | graph | 1.899 | 1.894–1.902 | 5 |
| tokens_8192 | 16 | raw | graph | 1.858 | 1.856–1.859 | 5 |
| tokens_8192 | 16 | tile16 | graph | 1.876 | 1.867–1.878 | 5 |
| tokens_8192 | 32 | center | graph | 1.919 | 1.917–1.923 | 5 |
| tokens_8192 | 32 | direct | graph | 2.042 | 2.035–2.043 | 5 |
| tokens_8192 | 32 | raw | graph | 1.920 | 1.913–1.922 | 5 |
| tokens_8192 | 32 | tile16 | graph | 2.487 | 2.481–2.492 | 5 |
| tokens_8192 | 64 | center | graph | 3.883 | 3.873–3.884 | 5 |
| tokens_8192 | 64 | direct | graph | 4.499 | 4.497–4.501 | 5 |
| tokens_8192 | 64 | raw | graph | 3.844 | 3.837–3.846 | 5 |
| tokens_8192 | 64 | tile16 | graph | 5.462 | 5.459–5.469 | 5 |

g=-5 的范围检查（每行包含完整 batch 的因果位置）：

| CHUNK | method | zero | Inf | NaN | max relative error |
|---:|---|---:|---:|---:|---:|
| 16 | raw | 0 | 0 | 0 | 0.0020976203408346003 |
| 16 | center | 0 | 0 | 0 | 0.0017519585915509105 |
| 16 | tile16 | 0 | 0 | 0 | 0.0017519585915509105 |
| 16 | direct | 0 | 0 | 0 | 6.233915444169894e-06 |
| 32 | raw | 2040 | 0 | 960 | nonfinite |
| 32 | center | 528 | 0 | 0 | 0.002247667027784329 |
| 32 | tile16 | 528 | 0 | 0 | 0.0017519580426397533 |
| 32 | direct | 840 | 0 | 0 | 6.336984365568169e-06 |
| 64 | raw | 6392 | 0 | 9024 | nonfinite |
| 64 | center | 10928 | 0 | 1680 | nonfinite |
| 64 | tile16 | 7568 | 0 | 0 | 0.0017519577681846944 |
| 64 | direct | 8648 | 0 | 0 | 6.3878951159923444e-06 |

## inverse

| workload | CHUNK | best candidate | µs/call | vs C=16 |
|---|---|---|---|---|
| single | 16 | unified_w8 | 2.881 | 1.00× |
| single | 32 | unified_w8 | 5.096 | 1.77× |
| single | 64 | unified_w8 | 32.225 | 11.18× |
| tokens_786432 | 16 | unified_w1 | 79.179 | 1.00× |
| tokens_786432 | 32 | unified_w4 | 314.198 | 3.97× |
| tokens_786432 | 64 | unified_w8 | 2208.534 | 27.89× |
| tokens_8192 | 16 | unified_w8 | 3.114 | 1.00× |
| tokens_8192 | 32 | unified_w8 | 6.530 | 2.10× |
| tokens_8192 | 64 | unified_w8 | 32.370 | 10.40× |

| workload | chunk | method | mode | µs/call | round range µs | rounds |
|---|---|---|---|---|---|---|
| single | 16 | official | graph | 2.933 | 2.923–2.937 | 5 |
| single | 16 | unified_w1 | graph | 4.467 | 4.458–4.470 | 5 |
| single | 16 | unified_w4 | graph | 3.107 | 3.101–3.113 | 5 |
| single | 16 | unified_w8 | graph | 2.881 | 2.874–2.882 | 5 |
| single | 32 | unified_w1 | graph | 13.671 | 13.667–13.676 | 5 |
| single | 32 | unified_w4 | graph | 5.936 | 5.933–5.948 | 5 |
| single | 32 | unified_w8 | graph | 5.096 | 5.087–5.103 | 5 |
| single | 64 | unified_w1 | graph | 102.212 | 102.210–102.254 | 5 |
| single | 64 | unified_w4 | graph | 34.961 | 34.948–34.967 | 5 |
| single | 64 | unified_w8 | graph | 32.225 | 32.220–32.238 | 5 |
| tokens_786432 | 16 | official | graph | 49.668 | 49.663–49.682 | 5 |
| tokens_786432 | 16 | unified_w1 | graph | 79.179 | 79.161–79.185 | 5 |
| tokens_786432 | 16 | unified_w4 | graph | 85.819 | 85.812–85.825 | 5 |
| tokens_786432 | 16 | unified_w8 | graph | 119.529 | 119.521–119.545 | 5 |
| tokens_786432 | 32 | unified_w1 | graph | 319.231 | 319.190–319.303 | 5 |
| tokens_786432 | 32 | unified_w4 | graph | 314.198 | 314.186–314.213 | 5 |
| tokens_786432 | 32 | unified_w8 | graph | 317.803 | 317.791–317.815 | 5 |
| tokens_786432 | 64 | unified_w1 | graph | 2475.995 | 2475.980–2476.252 | 5 |
| tokens_786432 | 64 | unified_w4 | graph | 2232.746 | 2232.609–2233.079 | 5 |
| tokens_786432 | 64 | unified_w8 | graph | 2208.534 | 2208.472–2208.558 | 5 |
| tokens_8192 | 16 | official | graph | 3.004 | 2.999–3.007 | 5 |
| tokens_8192 | 16 | unified_w1 | graph | 4.559 | 4.556–4.559 | 5 |
| tokens_8192 | 16 | unified_w4 | graph | 3.309 | 3.305–3.310 | 5 |
| tokens_8192 | 16 | unified_w8 | graph | 3.114 | 3.108–3.116 | 5 |
| tokens_8192 | 32 | unified_w1 | graph | 13.734 | 13.731–13.735 | 5 |
| tokens_8192 | 32 | unified_w4 | graph | 7.286 | 7.283–7.292 | 5 |
| tokens_8192 | 32 | unified_w8 | graph | 6.530 | 6.528–6.535 | 5 |
| tokens_8192 | 64 | unified_w1 | graph | 102.497 | 102.494–102.534 | 5 |
| tokens_8192 | 64 | unified_w4 | graph | 35.048 | 35.047–35.051 | 5 |
| tokens_8192 | 64 | unified_w8 | graph | 32.370 | 32.368–32.379 | 5 |


正确性指标：max_residual，下表跨种子取最大值。

| input/stage | CHUNK | method/warps | error |
|---|---:|---|---:|
| stress | 16 | official | 0.00156149 |
| stress | 16 | unified_w1 | 0.00156149 |
| stress | 16 | unified_w4 | 0.00156149 |
| stress | 16 | unified_w8 | 0.00156149 |
| stress | 32 | unified_w1 | 0.00241048 |
| stress | 32 | unified_w4 | 0.00241048 |
| stress | 32 | unified_w8 | 0.00241048 |
| stress | 64 | unified_w1 | 0.00513387 |
| stress | 64 | unified_w4 | 0.00513387 |
| stress | 64 | unified_w8 | 0.00513387 |
| structured | 16 | official | 0.00020329 |
| structured | 16 | unified_w1 | 0.00020329 |
| structured | 16 | unified_w4 | 0.00020329 |
| structured | 16 | unified_w8 | 0.00020329 |
| structured | 32 | unified_w1 | 0.000233312 |
| structured | 32 | unified_w4 | 0.000233312 |
| structured | 32 | unified_w8 | 0.000233312 |
| structured | 64 | unified_w1 | 0.000255041 |
| structured | 64 | unified_w4 | 0.000255041 |
| structured | 64 | unified_w8 | 0.000255041 |
| zero | 16 | official | 0 |
| zero | 16 | unified_w1 | 0 |
| zero | 16 | unified_w4 | 0 |
| zero | 16 | unified_w8 | 0 |
| zero | 32 | unified_w1 | 0 |
| zero | 32 | unified_w4 | 0 |
| zero | 32 | unified_w8 | 0 |
| zero | 64 | unified_w1 | 0 |
| zero | 64 | unified_w4 | 0 |
| zero | 64 | unified_w8 | 0 |

## mma

| workload | CHUNK | stage | best candidate | µs/call | vs C=16 |
|---|---|---|---|---|---|
| single | 16 | gram | wmma_w8 | 4.181 | 1.00× |
| single | 16 | local_mix | wmma_w8 | 2.803 | 1.00× |
| single | 16 | projection | wmma_w8 | 7.993 | 1.00× |
| single | 16 | state_update | wmma_w8 | 6.382 | 1.00× |
| single | 32 | gram | wmma_w4 | 4.945 | 1.18× |
| single | 32 | local_mix | wmma_w4 | 5.272 | 1.88× |
| single | 32 | projection | wmma_w8 | 11.613 | 1.45× |
| single | 32 | state_update | wmma_w4 | 9.186 | 1.44× |
| single | 64 | gram | wmma_w8 | 8.983 | 2.15× |
| single | 64 | local_mix | wmma_w8 | 10.644 | 3.80× |
| single | 64 | projection | wmma_w8 | 15.395 | 1.93× |
| single | 64 | state_update | wmma_w8 | 14.656 | 2.30× |
| tokens_786432 | 16 | gram | wmma_w1 | 234.357 | 1.00× |
| tokens_786432 | 16 | local_mix | wmma_w1 | 188.793 | 1.00× |
| tokens_786432 | 16 | projection | wmma_w8 | 1398.587 | 1.00× |
| tokens_786432 | 16 | state_update | wmma_w1 | 977.091 | 1.00× |
| tokens_786432 | 32 | gram | wmma_w4 | 249.698 | 1.07× |
| tokens_786432 | 32 | local_mix | wmma_w4 | 255.235 | 1.35× |
| tokens_786432 | 32 | projection | wmma_w8 | 1046.881 | 0.75× |
| tokens_786432 | 32 | state_update | wmma_w4 | 852.120 | 0.87× |
| tokens_786432 | 64 | gram | wmma_w8 | 482.288 | 2.06× |
| tokens_786432 | 64 | local_mix | wmma_w8 | 490.447 | 2.60× |
| tokens_786432 | 64 | projection | wmma_w8 | 916.337 | 0.66× |
| tokens_786432 | 64 | state_update | wmma_w8 | 927.621 | 0.95× |
| tokens_8192 | 16 | gram | wmma_w8 | 4.743 | 1.00× |
| tokens_8192 | 16 | local_mix | wmma_w8 | 3.942 | 1.00× |
| tokens_8192 | 16 | projection | wmma_w8 | 15.422 | 1.00× |
| tokens_8192 | 16 | state_update | wmma_w8 | 14.240 | 1.00× |
| tokens_8192 | 32 | gram | wmma_w4 | 5.993 | 1.26× |
| tokens_8192 | 32 | local_mix | wmma_w4 | 6.465 | 1.64× |
| tokens_8192 | 32 | projection | wmma_w8 | 16.683 | 1.08× |
| tokens_8192 | 32 | state_update | wmma_w4 | 13.532 | 0.95× |
| tokens_8192 | 64 | gram | wmma_w8 | 9.067 | 1.91× |
| tokens_8192 | 64 | local_mix | wmma_w8 | 10.730 | 2.72× |
| tokens_8192 | 64 | projection | wmma_w8 | 15.517 | 1.01× |
| tokens_8192 | 64 | state_update | wmma_w8 | 14.756 | 1.04× |

| workload | chunk | stage | method | mode | µs/call | round range µs | rounds |
|---|---|---|---|---|---|---|---|
| single | 16 | gram | wmma_w1 | graph | 7.100 | 7.100–7.113 | 5 |
| single | 16 | gram | wmma_w4 | graph | 6.829 | 6.828–6.836 | 5 |
| single | 16 | gram | wmma_w8 | graph | 4.181 | 4.165–4.182 | 5 |
| single | 16 | local_mix | wmma_w1 | graph | 6.038 | 6.035–6.056 | 5 |
| single | 16 | local_mix | wmma_w4 | graph | 4.477 | 4.469–4.479 | 5 |
| single | 16 | local_mix | wmma_w8 | graph | 2.803 | 2.801–2.816 | 5 |
| single | 16 | projection | wmma_w1 | graph | 32.784 | 32.779–32.797 | 5 |
| single | 16 | projection | wmma_w4 | graph | 12.221 | 12.220–12.223 | 5 |
| single | 16 | projection | wmma_w8 | graph | 7.993 | 7.982–7.995 | 5 |
| single | 16 | state_update | wmma_w1 | graph | 13.181 | 13.175–13.185 | 5 |
| single | 16 | state_update | wmma_w4 | graph | 9.305 | 9.291–9.307 | 5 |
| single | 16 | state_update | wmma_w8 | graph | 6.382 | 6.377–6.382 | 5 |
| single | 32 | gram | wmma_w1 | graph | 14.739 | 14.736–14.752 | 5 |
| single | 32 | gram | wmma_w4 | graph | 4.945 | 4.936–4.949 | 5 |
| single | 32 | gram | wmma_w8 | graph | 7.450 | 7.447–7.454 | 5 |
| single | 32 | local_mix | wmma_w1 | graph | 11.495 | 11.491–11.499 | 5 |
| single | 32 | local_mix | wmma_w4 | graph | 5.272 | 5.267–5.277 | 5 |
| single | 32 | local_mix | wmma_w8 | graph | 5.794 | 5.783–5.797 | 5 |
| single | 32 | projection | wmma_w1 | graph | 39.619 | 39.609–39.630 | 5 |
| single | 32 | projection | wmma_w4 | graph | 13.198 | 13.194–13.203 | 5 |
| single | 32 | projection | wmma_w8 | graph | 11.613 | 11.606–11.618 | 5 |
| single | 32 | state_update | wmma_w1 | graph | 23.908 | 23.903–23.925 | 5 |
| single | 32 | state_update | wmma_w4 | graph | 9.186 | 9.183–9.188 | 5 |
| single | 32 | state_update | wmma_w8 | graph | 11.329 | 11.323–11.334 | 5 |
| single | 64 | gram | wmma_w1 | graph | 34.856 | 34.840–34.861 | 5 |
| single | 64 | gram | wmma_w4 | graph | 11.851 | 11.838–11.853 | 5 |
| single | 64 | gram | wmma_w8 | graph | 8.983 | 8.976–8.985 | 5 |
| single | 64 | local_mix | wmma_w1 | graph | 30.196 | 30.192–30.210 | 5 |
| single | 64 | local_mix | wmma_w4 | graph | 10.902 | 10.901–10.908 | 5 |
| single | 64 | local_mix | wmma_w8 | graph | 10.644 | 10.641–10.653 | 5 |
| single | 64 | projection | wmma_w1 | graph | 56.462 | 56.457–56.464 | 5 |
| single | 64 | projection | wmma_w4 | graph | 19.860 | 19.857–19.865 | 5 |
| single | 64 | projection | wmma_w8 | graph | 15.395 | 15.387–15.398 | 5 |
| single | 64 | state_update | wmma_w1 | graph | 48.514 | 48.503–48.517 | 5 |
| single | 64 | state_update | wmma_w4 | graph | 17.610 | 17.588–17.620 | 5 |
| single | 64 | state_update | wmma_w8 | graph | 14.656 | 14.655–14.658 | 5 |
| tokens_786432 | 16 | gram | wmma_w1 | graph | 234.357 | 234.335–234.374 | 5 |
| tokens_786432 | 16 | gram | wmma_w4 | graph | 331.874 | 331.847–331.891 | 5 |
| tokens_786432 | 16 | gram | wmma_w8 | graph | 347.189 | 347.168–347.198 | 5 |
| tokens_786432 | 16 | local_mix | wmma_w1 | graph | 188.793 | 188.709–188.841 | 5 |
| tokens_786432 | 16 | local_mix | wmma_w4 | graph | 246.379 | 246.375–246.425 | 5 |
| tokens_786432 | 16 | local_mix | wmma_w8 | graph | 262.247 | 262.226–262.276 | 5 |
| tokens_786432 | 16 | projection | wmma_w1 | graph | 4104.196 | 4104.030–4104.560 | 5 |
| tokens_786432 | 16 | projection | wmma_w4 | graph | 1684.396 | 1684.314–1684.505 | 5 |
| tokens_786432 | 16 | projection | wmma_w8 | graph | 1398.587 | 1398.554–1398.611 | 5 |
| tokens_786432 | 16 | state_update | wmma_w1 | graph | 977.091 | 977.074–977.146 | 5 |
| tokens_786432 | 16 | state_update | wmma_w4 | graph | 995.652 | 995.600–995.737 | 5 |
| tokens_786432 | 16 | state_update | wmma_w8 | graph | 1020.986 | 1020.857–1020.997 | 5 |
| tokens_786432 | 32 | gram | wmma_w1 | graph | 482.981 | 482.917–483.237 | 5 |
| tokens_786432 | 32 | gram | wmma_w4 | graph | 249.698 | 249.695–249.708 | 5 |
| tokens_786432 | 32 | gram | wmma_w8 | graph | 412.892 | 412.832–412.926 | 5 |
| tokens_786432 | 32 | local_mix | wmma_w1 | graph | 313.605 | 313.562–313.668 | 5 |
| tokens_786432 | 32 | local_mix | wmma_w4 | graph | 255.235 | 255.199–255.241 | 5 |
| tokens_786432 | 32 | local_mix | wmma_w8 | graph | 344.223 | 344.206–344.260 | 5 |
| tokens_786432 | 32 | projection | wmma_w1 | graph | 3107.065 | 3106.965–3107.500 | 5 |
| tokens_786432 | 32 | projection | wmma_w4 | graph | 1160.117 | 1159.982–1160.330 | 5 |
| tokens_786432 | 32 | projection | wmma_w8 | graph | 1046.881 | 1046.770–1046.985 | 5 |
| tokens_786432 | 32 | state_update | wmma_w1 | graph | 958.908 | 958.861–958.962 | 5 |
| tokens_786432 | 32 | state_update | wmma_w4 | graph | 852.120 | 852.103–852.141 | 5 |
| tokens_786432 | 32 | state_update | wmma_w8 | graph | 905.026 | 904.991–905.093 | 5 |
| tokens_786432 | 64 | gram | wmma_w1 | graph | 1183.726 | 1183.673–1183.802 | 5 |
| tokens_786432 | 64 | gram | wmma_w4 | graph | 525.750 | 525.713–525.840 | 5 |
| tokens_786432 | 64 | gram | wmma_w8 | graph | 482.288 | 482.186–482.307 | 5 |
| tokens_786432 | 64 | local_mix | wmma_w1 | graph | 790.028 | 789.972–790.165 | 5 |
| tokens_786432 | 64 | local_mix | wmma_w4 | graph | 492.351 | 492.326–492.366 | 5 |
| tokens_786432 | 64 | local_mix | wmma_w8 | graph | 490.447 | 490.397–490.476 | 5 |
| tokens_786432 | 64 | projection | wmma_w1 | graph | 2609.124 | 2609.077–2609.457 | 5 |
| tokens_786432 | 64 | projection | wmma_w4 | graph | 1016.992 | 1016.902–1017.125 | 5 |
| tokens_786432 | 64 | projection | wmma_w8 | graph | 916.337 | 916.319–916.390 | 5 |
| tokens_786432 | 64 | state_update | wmma_w1 | graph | 1632.819 | 1632.775–1632.891 | 5 |
| tokens_786432 | 64 | state_update | wmma_w4 | graph | 937.745 | 937.737–937.815 | 5 |
| tokens_786432 | 64 | state_update | wmma_w8 | graph | 927.621 | 927.609–927.637 | 5 |
| tokens_8192 | 16 | gram | wmma_w1 | graph | 7.714 | 7.711–7.725 | 5 |
| tokens_8192 | 16 | gram | wmma_w4 | graph | 7.462 | 7.448–7.466 | 5 |
| tokens_8192 | 16 | gram | wmma_w8 | graph | 4.743 | 4.739–4.746 | 5 |
| tokens_8192 | 16 | local_mix | wmma_w1 | graph | 6.926 | 6.916–6.935 | 5 |
| tokens_8192 | 16 | local_mix | wmma_w4 | graph | 5.562 | 5.554–5.564 | 5 |
| tokens_8192 | 16 | local_mix | wmma_w8 | graph | 3.942 | 3.940–3.947 | 5 |
| tokens_8192 | 16 | projection | wmma_w1 | graph | 38.877 | 38.836–38.881 | 5 |
| tokens_8192 | 16 | projection | wmma_w4 | graph | 19.365 | 19.358–19.370 | 5 |
| tokens_8192 | 16 | projection | wmma_w8 | graph | 15.422 | 15.415–15.428 | 5 |
| tokens_8192 | 16 | state_update | wmma_w1 | graph | 18.837 | 18.833–18.848 | 5 |
| tokens_8192 | 16 | state_update | wmma_w4 | graph | 16.922 | 16.905–16.933 | 5 |
| tokens_8192 | 16 | state_update | wmma_w8 | graph | 14.240 | 14.237–14.246 | 5 |
| tokens_8192 | 32 | gram | wmma_w1 | graph | 15.468 | 15.456–15.470 | 5 |
| tokens_8192 | 32 | gram | wmma_w4 | graph | 5.993 | 5.988–6.004 | 5 |
| tokens_8192 | 32 | gram | wmma_w8 | graph | 8.574 | 8.569–8.577 | 5 |
| tokens_8192 | 32 | local_mix | wmma_w1 | graph | 11.988 | 11.983–11.996 | 5 |
| tokens_8192 | 32 | local_mix | wmma_w4 | graph | 6.465 | 6.462–6.467 | 5 |
| tokens_8192 | 32 | local_mix | wmma_w8 | graph | 6.910 | 6.908–6.916 | 5 |
| tokens_8192 | 32 | projection | wmma_w1 | graph | 43.680 | 43.658–43.693 | 5 |
| tokens_8192 | 32 | projection | wmma_w4 | graph | 18.175 | 18.171–18.179 | 5 |
| tokens_8192 | 32 | projection | wmma_w8 | graph | 16.683 | 16.678–16.700 | 5 |
| tokens_8192 | 32 | state_update | wmma_w1 | graph | 26.925 | 26.918–26.934 | 5 |
| tokens_8192 | 32 | state_update | wmma_w4 | graph | 13.532 | 13.525–13.533 | 5 |
| tokens_8192 | 32 | state_update | wmma_w8 | graph | 15.862 | 15.858–15.866 | 5 |
| tokens_8192 | 64 | gram | wmma_w1 | graph | 35.332 | 35.324–35.354 | 5 |
| tokens_8192 | 64 | gram | wmma_w4 | graph | 11.976 | 11.970–11.984 | 5 |
| tokens_8192 | 64 | gram | wmma_w8 | graph | 9.067 | 9.058–9.068 | 5 |
| tokens_8192 | 64 | local_mix | wmma_w1 | graph | 30.583 | 30.582–30.603 | 5 |
| tokens_8192 | 64 | local_mix | wmma_w4 | graph | 11.073 | 11.067–11.076 | 5 |
| tokens_8192 | 64 | local_mix | wmma_w8 | graph | 10.730 | 10.724–10.741 | 5 |
| tokens_8192 | 64 | projection | wmma_w1 | graph | 56.815 | 56.809–56.858 | 5 |
| tokens_8192 | 64 | projection | wmma_w4 | graph | 19.991 | 19.984–19.995 | 5 |
| tokens_8192 | 64 | projection | wmma_w8 | graph | 15.517 | 15.515–15.518 | 5 |
| tokens_8192 | 64 | state_update | wmma_w1 | graph | 48.739 | 48.733–48.744 | 5 |
| tokens_8192 | 64 | state_update | wmma_w4 | graph | 17.845 | 17.841–17.851 | 5 |
| tokens_8192 | 64 | state_update | wmma_w8 | graph | 14.756 | 14.750–14.760 | 5 |


正确性指标：max_relative_error，下表跨种子取最大值。

| input/stage | CHUNK | method/warps | error |
|---|---:|---|---:|
| gram | 16 | 1 | 1.39613e-07 |
| gram | 16 | 4 | 1.39613e-07 |
| gram | 16 | 8 | 1.39613e-07 |
| gram | 32 | 1 | 1.23872e-07 |
| gram | 32 | 4 | 1.23872e-07 |
| gram | 32 | 8 | 1.23872e-07 |
| gram | 64 | 1 | 1.21238e-07 |
| gram | 64 | 4 | 1.21238e-07 |
| gram | 64 | 8 | 1.21238e-07 |
| local_mix | 16 | 1 | 3.23404e-08 |
| local_mix | 16 | 4 | 3.23404e-08 |
| local_mix | 16 | 8 | 3.23404e-08 |
| local_mix | 32 | 1 | 4.28524e-08 |
| local_mix | 32 | 4 | 4.28524e-08 |
| local_mix | 32 | 8 | 4.28524e-08 |
| local_mix | 64 | 1 | 6.89952e-08 |
| local_mix | 64 | 4 | 6.89952e-08 |
| local_mix | 64 | 8 | 6.89952e-08 |
| projection | 16 | 1 | 1.29512e-07 |
| projection | 16 | 4 | 1.29512e-07 |
| projection | 16 | 8 | 1.29512e-07 |
| projection | 32 | 1 | 1.28012e-07 |
| projection | 32 | 4 | 1.28012e-07 |
| projection | 32 | 8 | 1.28012e-07 |
| projection | 64 | 1 | 1.2419e-07 |
| projection | 64 | 4 | 1.2419e-07 |
| projection | 64 | 8 | 1.2419e-07 |
| state_update | 16 | 1 | 2.64731e-08 |
| state_update | 16 | 4 | 2.64731e-08 |
| state_update | 16 | 8 | 2.64731e-08 |
| state_update | 32 | 1 | 4.09734e-08 |
| state_update | 32 | 4 | 4.09734e-08 |
| state_update | 32 | 8 | 4.09734e-08 |
| state_update | 64 | 1 | 6.73107e-08 |
| state_update | 64 | 4 | 6.73107e-08 |
| state_update | 64 | 8 | 6.73107e-08 |

## 求逆资源

| C | warps | registers/thread | shared bytes | local bytes/thread | resident CTA/SM max |
|---:|---:|---:|---:|---:|---:|
| 16 | 1 | 31 | 3072 | 0 | 32 |
| 16 | 4 | 31 | 6144 | 0 | 16 |
| 16 | 8 | 31 | 10240 | 0 | 8 |
| 32 | 1 | 39 | 9216 | 0 | 22 |
| 32 | 4 | 32 | 12288 | 0 | 16 |
| 32 | 8 | 32 | 16384 | 0 | 8 |
| 64 | 1 | 40 | 33792 | 0 | 6 |
| 64 | 4 | 40 | 36864 | 0 | 6 |
| 64 | 8 | 40 | 40960 | 0 | 5 |

## 时钟与检查

- inverse_clocks.csv: 130 samples, 2032–2032 MHz。
- mma_clocks.csv: 527 samples, 2032–2032 MHz。
- range_clocks.csv: 7 samples, 2032–2032 MHz。
- racecheck: passed。

## 解释边界

- 所有微基准是预分配缓冲区、重复输入的局部算子测量，不是完整 KDA 或 K1/K2 耗时。
- range 的 tile16 只修复因果衰减矩阵的构造，尚未整合进 KDA；direct 是稳定参考，不是推荐实现。
- 官方求逆与统一 WMMA 的累加精度和融合组织不同，官方只作 16×16 参照。
- 最佳配置只在本次 1/4/8 warp 候选内选择，不代表最优实现或算法下界。
- MMA 是独立矩阵，不模拟 K2 的状态依赖；不能将四类耗时直接相加当作端到端时间。
