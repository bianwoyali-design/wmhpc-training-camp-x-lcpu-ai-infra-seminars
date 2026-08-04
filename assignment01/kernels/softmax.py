"""问题 7.8（选做）：softmax in Triton（FROM-SCRATCH）。

注：此题可以不用GPU (conftest.py 会自动切到 interpreter 模式)。

contract：
- softmax(x) 接收形状 (M, N) 的 2D tensor，返回同形状结果，
  对每一行独立做 softmax；
- kernel 自己写，一个 program 处理一行；
- 为了确保数值稳定，要求行内先减最大值，再做 exp 与求和。测试里有一行
  数值巨大的输入，不稳定的实现会得到 inf/nan；
- 行宽 N 任意（用 mask 处理），可以假设 N <= 4096，BLOCK_SIZE 用
  triton.next_power_of_2(N) 是常见做法；
- 通过 pytest tests/test_softmax.py 即为完成。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    a_ptr,
    b_ptr,
    N,
    stride_am,
    stride_an,
    stride_bm,
    stride_bn,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)

    offs_n = tl.arange(0, BLOCK_SIZE)
    mask = offs_n < N

    a_ptrs = a_ptr + pid_m * stride_am + offs_n * stride_an
    b_ptrs = b_ptr + pid_m * stride_bm + offs_n * stride_bn

    a = tl.load(
        a_ptrs,
        mask=mask,
        other=-float("inf"),
    )

    a = a - tl.max(a, axis=0)

    numerator = tl.exp(a)
    denominator = tl.sum(numerator, axis=0)
    b = numerator / denominator

    tl.store(
        b_ptrs,
        b,
        mask=mask,
    )


def softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float32
    assert x.ndim == 2

    M, N = x.shape
    assert N <= 4096

    y = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)

    softmax_kernel[grid](
        x,
        y,
        N,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y
