"""CPU FP64 token recurrence; checkpoint rounding is the only model ablation."""

import importlib.util
from pathlib import Path

import torch


@torch.jit.script
def recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial: torch.Tensor,
    points: list[int],
):
    # q/k already normalized, q already scaled. State layout is [K,V].
    # Modes: exact FP64, FP64 + BF16 checkpoint, FP64 + FP32 checkpoint.
    state = initial.unsqueeze(0).repeat(3, 1, 1, 1)
    output = torch.empty((3, q.size(0), q.size(1), v.size(2)), dtype=q.dtype)
    snapshots = torch.empty(
        (3, len(points), q.size(1), k.size(2), v.size(2)), dtype=q.dtype
    )
    decay = g.exp()
    cursor = 0
    for t in range(q.size(0)):
        state = state * decay[t].unsqueeze(0).unsqueeze(-1)
        residual = v[t].unsqueeze(0) - (k[t].unsqueeze(0).unsqueeze(-1) * state).sum(-2)
        state = state + (beta[t].unsqueeze(-1) * k[t]).unsqueeze(0).unsqueeze(
            -1
        ) * residual.unsqueeze(-2)
        output[:, t] = (q[t].unsqueeze(0).unsqueeze(-1) * state).sum(-2)
        if (t + 1) % 16 == 0 or t + 1 == q.size(0):
            state[1] = state[1].to(torch.bfloat16).to(torch.float64)
            state[2] = state[2].to(torch.float32).to(torch.float64)
        if cursor < len(points) and t + 1 == points[cursor]:
            snapshots[:, cursor] = state
            cursor += 1
    return output, snapshots


def activate(q, k, v, g, beta, initial, alog, bias):
    q = q.double()
    k = k.double()
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    gate = -5 * torch.sigmoid(
        alog.double().exp()[None, :, None] * (g.double() + bias.double()[None])
    )
    return (
        q * 128**-0.5,
        k,
        v.double(),
        gate,
        torch.sigmoid(beta.double()),
        initial.double().transpose(-1, -2).contiguous(),
    )


def validate():
    path = Path(__file__).resolve().parents[2] / "fla_kda_ref/naive.py"
    spec = importlib.util.spec_from_file_location("provided_naive", path)
    naive = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(naive)
    rows = []
    for seed in [0, 1, 2]:
        for t in [1, 15, 16, 17, 33]:
            torch.manual_seed(seed)
            q = torch.randn(t, 2, 128).double()
            k = torch.randn_like(q)
            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)
            v = torch.randn_like(q)
            g = -torch.rand_like(q) * 0.1
            beta = torch.sigmoid(torch.randn(t, 2).double())
            initial = torch.randn(2, 128, 128).double() * 0.1
            out, states = recurrence(q * 128**-0.5, k, v, g, beta, initial, [t])
            gold, final = naive.naive_recurrent_kda(
                q[None],
                k[None],
                v[None],
                g[None],
                beta[None],
                initial_state=initial[None],
                output_final_state=True,
            )
            oe = ((out[0] - gold[0]).norm() / gold.norm()).item()
            se = ((states[0, 0] - final[0]).norm() / final.norm()).item()
            assert max(oe, se) < 2e-5, (seed, t, oe, se)
            rows.append(
                dict(seed=seed, length=t, output_relative=oe, state_relative=se)
            )
    return rows
