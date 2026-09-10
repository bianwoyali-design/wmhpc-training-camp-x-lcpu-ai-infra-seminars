"""CPU numerical diagnostics, not a FlashKDA correctness or speed benchmark."""

import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flashkda-range-mpl")
import torch

PATHS = ["fp64", "fp32", "fp32_bf16", "ftz_model_bf16"]


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    ftz_control_supported = torch.set_flush_denormal(False)
    rows, traces = [], []
    for seed in range(5):
        for case, full in inputs(seed).items():
            # Constants need no repeated seeds.
            if seed and case.startswith("constant"):
                continue
            for chunk in [16, 32, 64]:
                ref = evaluate(full[:chunk], "fp64")
                assert torch.allclose(
                    ref["direct"], ref["factorized"], rtol=1e-12, atol=1e-300
                )
                for path in PATHS:
                    result = evaluate(full[:chunk], path)
                    for quantity, x in result.items():
                        # Both ratio expressions are compared to direct FP64, not
                        # to an independently rounded factorized reference.
                        gold = (
                            ref["direct"] if quantity == "factorized" else ref[quantity]
                        )
                        rows.append(
                            dict(
                                chunk=chunk,
                                gate_case=case,
                                seed=seed,
                                numeric_path=path,
                                quantity=quantity,
                                **stats(x, gold, quantity),
                            )
                        )
                    if chunk == 64:
                        for i in range(chunk):
                            traces.append(
                                dict(
                                    gate_case=case,
                                    seed=seed,
                                    numeric_path=path,
                                    token=i + 1,
                                    gate=full[i].item(),
                                    prefix=result["prefix"][i].item(),
                                    decay=result["decay"][i].item(),
                                    inverse=result["inverse"][i].item(),
                                )
                            )
                    if case == "constant_0":
                        assert torch.equal(
                            result["direct"], torch.ones_like(result["direct"])
                        )
                        assert torch.equal(
                            result["factorized"], torch.ones_like(result["factorized"])
                        )
    for filename, data in [("metrics.csv", rows), ("traces.csv", traces)]:
        with (args.out / filename).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    metadata = dict(
        torch=torch.__version__,
        device="cpu",
        seeds=list(range(5)),
        chunks=[16, 32, 64],
        paths=PATHS,
        metrics_rows=len(rows),
        flush_denormal_disable_supported=ftz_control_supported,
        note="No CUDA instruction execution; FTZ model only. No KDA output/state validation.",
    )
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot(args.out, rows, traces)
    print(json.dumps(metadata, indent=2))
    for r in rows:
        if (
            r["gate_case"] == "constant_-5"
            and r["numeric_path"] == "ftz_model_bf16"
            and r["quantity"] in ["decay", "inverse", "factorized"]
        ):
            print(r)


def plot(out, rows, traces):
    # Chart contract: static PNG/SVG; 64 ordered token observations; log10 decay
    # plus lost-value markers. Blue + orange, differentiated markers/line styles.
    # Error comparison: discrete chunk sizes, grouped bars; nonfinite cases
    # labelled explicitly instead of excluded silently. Source: generated CSVs.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ref = [
        r
        for r in traces
        if r["gate_case"] == "constant_-5" and r["numeric_path"] == "fp64"
    ]
    ax.plot(
        [r["token"] for r in ref],
        [r["prefix"] / math.log(10) for r in ref],
        color="#2458a6",
        label="FP64 reference",
    )
    for path, color, marker in [
        ("fp32_bf16", "#d17724", "o"),
        ("ftz_model_bf16", "#333333", "x"),
    ]:
        data = [
            r
            for r in traces
            if r["gate_case"] == "constant_-5" and r["numeric_path"] == path
        ]
        good = [r for r in data if r["decay"] > 0]
        ax.plot(
            [r["token"] for r in good],
            [math.log10(r["decay"]) for r in good],
            linestyle="--",
            marker=marker,
            markersize=3,
            color=color,
            label=path,
        )
        first = next((r["token"] for r in data if r["decay"] == 0), None)
        if first:
            ax.axvline(first, color=color, linestyle=":", alpha=0.7)
            ax.text(
                first + 1,
                -100 if path == "fp32_bf16" else -120,
                f"first zero: {first}",
                color=color,
            )
    ax.set(
        title="Decay versus token position\nActivated gate = -5; CPU diagnostic paths; 1-based token index",
        xlabel="Token position",
        ylabel="log10(decay), positive values only",
    )
    ax.grid(alpha=0.15)
    ax.legend(loc="lower left")
    for ext in ["png", "svg"]:
        fig.savefig(out / f"decay.{ext}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained")
    for ax, case in zip(axes, ["constant_-1", "constant_-5"]):
        for j, (quantity, color) in enumerate(
            [("direct", "#2458a6"), ("factorized", "#d17724")]
        ):
            data = [
                next(
                    r
                    for r in rows
                    if r["chunk"] == c
                    and r["gate_case"] == case
                    and r["numeric_path"] == "ftz_model_bf16"
                    and r["quantity"] == quantity
                )
                for c in [16, 32, 64]
            ]
            for i, r in enumerate(data):
                x = i + (j - 0.5) * 0.34
                if r["relative_l2_error"] is None:
                    ax.text(
                        x,
                        0.0001,
                        "nonfinite",
                        rotation=90,
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )
                else:
                    ax.bar(
                        x,
                        r["relative_l2_error"],
                        width=0.3,
                        color=color,
                        label=quantity if i == 0 else None,
                        hatch="//" if j else None,
                    )
        ax.set_xticks(range(3), [16, 32, 64])
        ax.set_xlabel("CHUNK")
        ax.set_title(case)
        ax.set_ylim(0, 0.0028)
        ax.set_xlim(-0.6, 2.6)
        ax.grid(axis="y", alpha=0.15)
        ax.legend(loc="upper right", fontsize=9)
    axes[0].set_ylabel("Relative L2 error vs FP64 direct ratio")
    fig.suptitle(
        "Causal decay ratio error\nCPU FTZ-output model + BF16 storage; all j <= i pairs"
    )
    for ext in ["png", "svg"]:
        fig.savefig(out / f"ratio_error.{ext}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
