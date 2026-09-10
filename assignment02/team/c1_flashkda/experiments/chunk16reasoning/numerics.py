"""CPU numerical diagnostics, not a FlashKDA correctness or speed benchmark."""

import csv
import json
import math
from pathlib import Path

import torch


def inputs(seed):
    rng = torch.Generator().manual_seed(seed)
    cases = {
        f"constant_{g:g}": torch.full((64,), g, dtype=torch.float64)
        for g in [0.0, -0.1, -1.0, -3.0, -5.0]
    }
    cases["uniform"] = -5 * torch.rand(64, generator=rng, dtype=torch.float64)
    cases["mixed"] = torch.where(
        torch.rand(64, generator=rng) < 0.1, -5.0, -0.01
    ).double()
    return cases


def evaluate(g, path):
    # Sequential accumulation; synthetic activated gates use natural-log units.
    dtype = torch.float64 if path == "fp64" else torch.float32
    values = g.to(dtype)
    if path == "ftz_model_bf16":
        values = values * math.log2(math.e)
    acc = torch.zeros((), dtype=dtype)
    prefix = []
    for value in values:
        acc = acc + value
        prefix.append(acc)
    s = torch.stack(prefix)

    def exponential(x):
        y = torch.exp2(x) if path == "ftz_model_bf16" else torch.exp(x)
        if path == "ftz_model_bf16":
            # Output FTZ model only: no PTX approximate-instruction emulation.
            y = torch.where(y.abs() < torch.finfo(torch.float32).tiny, 0.0, y)
        if path.endswith("bf16"):
            y = y.bfloat16().float()
        return y

    d, u = exponential(s), exponential(-s)
    mask = torch.ones((len(g), len(g)), dtype=torch.bool).tril()
    direct = exponential(s[:, None] - s[None, :])[mask]
    factorized = (d[:, None] * u[None, :])[mask]
    return {
        "prefix": s / math.log2(math.e) if path == "ftz_model_bf16" else s,
        "decay": d,
        "inverse": u,
        "direct": direct,
        "factorized": factorized,
    }


def stats(x, ref, quantity):
    x, ref = x.double(), ref.double()
    finite = torch.isfinite(x)
    zero_loss = (x == 0) & (ref != 0)
    bad = ~finite | zero_loss
    pos = torch.nonzero(bad).flatten()
    # Do not silently omit nonfinite values from error metrics.
    if finite.all():
        delta = x - ref
        max_abs = delta.abs().max().item()
        rel = (
            torch.linalg.vector_norm(delta)
            / torch.linalg.vector_norm(ref).clamp_min(1e-300)
        ).item()
    else:
        max_abs = rel = None
    selected = x[finite]
    return dict(
        zero_count=int((x == 0).sum()),
        lost_nonzero_count=int(zero_loss.sum()),
        inf_count=int(torch.isinf(x).sum()),
        nan_count=int(torch.isnan(x).sum()),
        first_bad_flat_index=int(pos[0]) if len(pos) else None,
        first_bad_token_1based=int(pos[0]) + 1
        if len(pos) and quantity in ["prefix", "decay", "inverse"]
        else None,
        finite_min=selected.min().item() if len(selected) else None,
        finite_max=selected.max().item() if len(selected) else None,
        max_abs_error=max_abs,
        relative_l2_error=rel,
        count=x.numel(),
    )


def cases():
    return [
        (name, seed, g)
        for seed in range(5)
        for name, g in inputs(seed).items()
        if not (seed and name.startswith("constant"))
    ]


def compare(out, mode):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    data = cases()
    torch.set_flush_denormal(False)
    if mode == "prepare":
        (out / "inputs.txt").write_text(
            str(len(data))
            + "\n"
            + "\n".join(
                " ".join(format(x, ".9g") for x in g.float().tolist())
                for _, _, g in data
            )
            + "\n"
        )
        (out / "inputs.json").write_text(
            json.dumps(
                [
                    dict(input=i, gate_case=n, seed=s)
                    for i, (n, s, g) in enumerate(data)
                ],
                indent=2,
            )
            + "\n"
        )
        return
    traces = list(csv.DictReader((out / "gpu_traces.csv").open()))
    pairs = list(csv.DictReader((out / "gpu_pairs.csv").open()))
    rows = []
    for idx, (name, seed, g) in enumerate(data):
        for chunk in [16, 32, 64]:
            ts = [
                r for r in traces if int(r["input"]) == idx and int(r["chunk"]) == chunk
            ]
            ps = [
                r for r in pairs if int(r["input"]) == idx and int(r["chunk"]) == chunk
            ]
            assert len(ts) == chunk and len(ps) == chunk * (chunk + 1) // 2
            # Nine significant digits round-trip FP32; restore FP32 first so
            # decimal serialization noise is not mistaken for GPU error.
            gpu = {
                q: torch.tensor(
                    [
                        float(r[q])
                        for r in (ps if q in ["direct", "factorized"] else ts)
                    ],
                    dtype=torch.float32,
                ).double()
                for q in ["decay", "inverse", "direct", "factorized"]
            }
            model = evaluate(g[:chunk], "ftz_model_bf16")
            ref = evaluate(g[:chunk], "fp64")
            for q, x in gpu.items():
                gold = ref["direct"] if q == "factorized" else ref[q]
                m = model[q].double()
                finite = torch.isfinite(x) & torch.isfinite(m)
                row = dict(
                    gate_case=name,
                    seed=seed,
                    chunk=chunk,
                    quantity=q,
                    **stats(x, gold, q),
                )
                row.update(
                    model_zero_mask_mismatches=int(((x == 0) != (m == 0)).sum()),
                    model_inf_mask_mismatches=int(
                        (torch.isinf(x) != torch.isinf(m)).sum()
                    ),
                    model_nan_mask_mismatches=int(
                        (torch.isnan(x) != torch.isnan(m)).sum()
                    ),
                    model_finite_max_abs_difference=(x[finite] - m[finite])
                    .abs()
                    .max()
                    .item()
                    if finite.any()
                    else None,
                )
                rows.append(row)
                if name == "constant_0":
                    assert torch.equal(x, torch.ones_like(x))
                if name == "constant_-5" and chunk >= 32 and q in ["decay", "inverse"]:
                    assert row["first_bad_token_1based"] == 18
    with (out / "comparison.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    summary = dict(
        rows=len(rows),
        torch=torch.__version__,
        zero_mask_mismatches=sum(r["model_zero_mask_mismatches"] for r in rows),
        inf_mask_mismatches=sum(r["model_inf_mask_mismatches"] for r in rows),
        nan_mask_mismatches=sum(r["model_nan_mask_mismatches"] for r in rows),
        finite_difference_rows=sum(
            r["model_finite_max_abs_difference"] not in [None, 0] for r in rows
        ),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

