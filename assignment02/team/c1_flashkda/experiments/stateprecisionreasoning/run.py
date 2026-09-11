"""Discussion 5: state precision, long memory, API dtype, and streaming checks."""

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from reference import activate, recurrence, validate

ROOT = Path(__file__).resolve().parent
CASES = (
    [
        (f"{p}_{t}", p, t)
        for p in ["nominal", "weak", "near_identity"]
        for t in [256, 8192, 65536]
    ]
    + [
        ("nonzero_8192", "nonzero", 8192),
        ("correlated_8192", "correlated", 8192),
        ("dynamic_8192", "dynamic", 8192),
    ]
    + [(f"retention_{t}", "retention", t) for t in [256, 8192, 65536]]
    + [("varlen", "weak", 1456)]
)


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def csvwrite(path, rows):
    if not rows:
        return
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def make_inputs(pattern, t, varlen=False):
    seeds = [0] if pattern == "retention" else [0, 1, 2]
    lengths = [15, 16, 17, 127, 128, 129, 1024] if varlen else [t]
    arrays = []
    for seed in seeds:
        gen = torch.Generator().manual_seed(seed)
        q = torch.randn(t, 128, generator=gen)
        k = torch.randn(t, 128, generator=gen)
        v = torch.randn(t, 128, generator=gen)
        g = torch.randn(t, 128, generator=gen)
        beta = torch.randn(t, generator=gen)
        init = 0.1 * torch.randn(len(lengths), 128, 128, generator=gen)
        if pattern == "nominal":
            init.zero_()
        if pattern in ["weak", "correlated"]:
            g.fill_(-8)
        if pattern == "near_identity":
            g.fill_(-16)
            beta.fill_(-4)
        if pattern == "correlated":
            key = torch.randn(1, 128, generator=gen)
            q = key + 0.01 * q
            k = key + 0.01 * k
            beta.fill_(4)
        if pattern == "dynamic":
            scale = 2.0 ** torch.linspace(-8, 8, 128)
            v *= scale
            init *= scale[:, None]
            g = torch.tensor([-16.0, -8.0, 0.0, 8.0]).repeat(32).expand(t, 128).clone()
        if pattern == "retention":
            q.zero_()
            k.zero_()
            q[:, 0] = 1
            k[:, 0] = 1
            v.zero_()
            g.fill_(-16)
            beta.fill_(-12)
            init.zero_()
            init[:, :, 0] = 1
        arrays.append((q, k, v, g, beta, init))
    tensors = [torch.stack([a[i] for a in arrays], dim=1).bfloat16() for i in range(5)]
    # Keep FP32 initial values for the API conversion control; reference uses BF16-rounded values.
    initial = torch.stack([a[5] for a in arrays], dim=1)
    h = len(seeds)
    return tensors, initial, torch.zeros(h), torch.zeros(h, 128), lengths, seeds


def flash(gpu, initial, alog, bias, start=0, end=None, lengths=None, fp32=False):
    import flash_kda_C as ext

    end = gpu[0].shape[0] if end is None else end
    q, k, v, g, beta = [a[start:end].contiguous().unsqueeze(0) for a in gpu]
    # A contiguous H=1 slice can retain an unaligned storage offset; TMA needs 16-byte alignment.
    beta = beta.clone()
    h = q.shape[2]
    n = initial.shape[0]
    init = initial.to(torch.float32 if fp32 else torch.bfloat16).contiguous()
    final = torch.empty_like(init)
    out = torch.empty_like(v)
    ws = torch.empty(
        ext.get_workspace_size(end - start, h, n), device="cuda", dtype=torch.uint8
    )
    cu = (
        torch.tensor(
            [0] + list(__import__("itertools").accumulate(lengths)),
            device="cuda",
            dtype=torch.int64,
        )
        if lengths
        else None
    )
    ext.fwd(q, k, v, g, beta, 128**-0.5, out, ws, alog, bias, -5.0, init, final, cu)
    return out[0], final


def metrics(actual, gold):
    a = actual.double().reshape(-1, actual.shape[-1])
    b = gold.double().reshape_as(a)
    d = a - b
    absd = d.abs()
    rms = b.square().mean().sqrt().item()
    rmse = d.square().mean().sqrt().item()
    per_channel = d.square().sum(0).sqrt() / b.square().sum(0).sqrt().clamp_min(1e-30)
    return dict(
        finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        ref_rms=rms,
        rmse=rmse,
        relative_l2=rmse / max(rms, 1e-30),
        max_abs=absd.max().item(),
        p99_abs=torch.quantile(absd.flatten(), 0.99).item(),
        worst_channel_relative=per_channel.max().item(),
    )


def analyze(name, pattern, t, tables):
    raw, initial, alog, bias, lengths, seeds = make_inputs(pattern, t, name == "varlen")
    gpu = [a.cuda() for a in raw]
    gi = initial.cuda()
    ga = alog.cuda()
    gb = bias.cuda()
    official, final = flash(
        gpu, gi, ga, gb, lengths=lengths if name == "varlen" else None
    )
    fpout, fpstate = flash(
        gpu, gi, ga, gb, lengths=lengths if name == "varlen" else None, fp32=True
    )
    equal_o = torch.equal(official, fpout)
    equal_s = torch.equal(final.float(), fpstate)
    assert equal_o and equal_s, (name, "API dtype control")
    tables["checks"].append(
        dict(
            case=name,
            test="fp32_api_vs_bf16_api",
            output_exact=equal_o,
            state_exact=equal_s,
            output_relative=0.0,
            state_relative=0.0,
        )
    )
    official = official.cpu().double()
    final = final.cpu().double()
    start = 0
    for seq, length in enumerate(lengths):
        points = sorted(
            set(
                [
                    x
                    for x in [16, 64, 256, 1024, 4096, 8192, 16384, 32768, 65536]
                    if x <= length
                ]
                + [length]
            )
        )
        args = activate(
            *[a[start : start + length] for a in raw],
            initial[seq].bfloat16(),
            alog,
            bias,
        )
        outputs, states = recurrence(*args, points)
        if name == "varlen":
            separate, separate_state = flash(
                gpu, gi[seq : seq + 1], ga, gb, start, start + length
            )
            eo = torch.equal(separate.cpu().double(), official[start : start + length])
            es = torch.equal(separate_state[0].cpu().double(), final[seq])
            assert eo and es, (name, seq, "varlen vs separate")
            tables["checks"].append(
                dict(
                    case=name,
                    test=f"varlen_seq_{seq}",
                    output_exact=eo,
                    state_exact=es,
                    output_relative=0.0,
                    state_relative=0.0,
                )
            )
        actuals = {
            "official": official[start : start + length],
            "bf16_checkpoint": outputs[1],
            "fp32_checkpoint": outputs[2],
            "output_rounding_floor": outputs[0].bfloat16().double(),
        }
        for method, actual in actuals.items():
            for h, seed in enumerate(seeds):
                row = dict(
                    case=name,
                    pattern=pattern,
                    length=length,
                    sequence=seq,
                    seed=seed,
                    method=method,
                    **metrics(actual[:, h], outputs[0, :, h]),
                )
                # Predeclared screening threshold, not a downstream model-quality guarantee.
                row["within_2pct"] = row["finite"] and row["relative_l2"] <= 0.02
                tables["outputs"].append(row)
                for lo in range(0, length, 256):
                    hi = min(lo + 256, length)
                    tables["windows"].append(
                        dict(
                            case=name,
                            sequence=seq,
                            seed=seed,
                            method=method,
                            start=lo,
                            end=hi,
                            **metrics(actual[lo:hi, h], outputs[0, lo:hi, h]),
                        )
                    )
        for pos, point in enumerate(points):
            if point == length:
                observed = final[seq]
            else:
                _, prefix_state = flash(
                    gpu, gi[seq : seq + 1], ga, gb, start, start + point
                )
                observed = prefix_state[0].cpu().double()
            candidates = {
                "official": observed.transpose(-1, -2),
                "bf16_checkpoint": states[1, pos],
                "fp32_checkpoint": states[2, pos],
            }
            for method, actual in candidates.items():
                for h, seed in enumerate(seeds):
                    row = dict(
                        case=name,
                        pattern=pattern,
                        length=length,
                        sequence=seq,
                        seed=seed,
                        position=point,
                        method=method,
                        **metrics(actual[h], states[0, pos, h]),
                    )
                    row["within_2pct"] = row["finite"] and row["relative_l2"] <= 0.02
                    tables["states"].append(row)
        if pattern == "retention":
            q, k, v, g, beta, init = args
            factor = g[0, 0, 0].exp() * (1 - beta[0, 0] * k[0, 0, 0] ** 2)
            analytic = init[0, 0, 0] * factor**length
            error = abs(states[0, -1, 0, 0, 0].item() - analytic.item())
            assert error < 1e-10, (name, error)
            tables["retention"].append(
                dict(
                    case=name,
                    length=length,
                    analytic_state=analytic.item(),
                    reference_state=states[0, -1, 0, 0, 0].item(),
                    official_state=final[0, 0, 0, 0].item(),
                    bf16_checkpoint_state=states[1, -1, 0, 0, 0].item(),
                    fp32_checkpoint_state=states[2, -1, 0, 0, 0].item(),
                    analytic_abs_error=error,
                )
            )
        start += length
    if len(lengths) == 1 and (t == 8192 or pattern == "retention" and t == 65536):
        for segment in [256, 1024, 17]:
            # 17 tests an unaligned first boundary; the rest is one complete call.
            bounds = list(range(0, t, segment)) + [t] if segment != 17 else [0, 17, t]
            state = gi
            parts = []
            for lo, hi in zip(bounds, bounds[1:]):
                o, state = flash(gpu, state, ga, gb, lo, hi)
                parts.append(o)
            streamed = torch.cat(parts).cpu().double()
            state = state.cpu().double()
            eo = torch.equal(streamed, official)
            es = torch.equal(state, final)
            if segment != 17:
                assert eo and es, (name, segment, "aligned streaming")
            tables["checks"].append(
                dict(
                    case=name,
                    test=f"stream_{segment}",
                    output_exact=eo,
                    state_exact=es,
                    output_relative=metrics(streamed, official)["relative_l2"],
                    state_relative=metrics(state, final)["relative_l2"],
                )
            )
    del gpu, official, final, fpout, fpstate
    torch.cuda.empty_cache()


def report(out):
    rows = list(csv.DictReader((out / "outputs.csv").open()))
    states = list(csv.DictReader((out / "states.csv").open()))
    lines = [
        "# 状态精度结果",
        "",
        "相对误差为 L2(error)/L2(FP64 reference)，下表取各 seed/sequence 的最大值，单位 %。",
        "BF16/FP32 checkpoint 是只改变每 16 token 状态舍入的 FP64 模型，不是官方 API dtype。",
        "",
        "| case | official output | official final state | BF16 checkpoint output | BF16 checkpoint final state |",
        "|---|---:|---:|---:|---:|",
    ]
    for case in dict.fromkeys(r["case"] for r in rows):
        values = []
        for method, kind in [
            ("official", "output"),
            ("official", "state"),
            ("bf16_checkpoint", "output"),
            ("bf16_checkpoint", "state"),
        ]:
            source = (
                rows
                if kind == "output"
                else [r for r in states if r["position"] == r["length"]]
            )
            values.append(
                max(
                    float(r["relative_l2"])
                    for r in source
                    if r["case"] == case and r["method"] == method
                )
                * 100
            )
        lines.append(
            "| " + case + " | " + " | ".join(f"{v:.4f}" for v in values) + " |"
        )
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["reproduce", "check", "report"])
    p.add_argument("--out", type=Path)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.command == "check":
        print(json.dumps(validate(), indent=2))
        return
    if args.command == "report":
        report(args.out)
        return
    import flash_kda_C

    if torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("需要 B300")
    out = (
        args.out
        or ROOT / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    ).resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = [CASES[0], CASES[-4]] if args.quick else CASES
    meta = dict(
        status="running",
        job=os.getenv("SLURM_JOB_ID"),
        device=str(torch.cuda.get_device_properties(0)),
        uuid=str(torch.cuda.get_device_properties(0).uuid),
        torch=torch.__version__,
        cases=cases,
        sources={p.name: digest(p) for p in [ROOT / "run.py", ROOT / "reference.py"]},
        extension_sha256=digest(flash_kda_C.__file__),
        screening_relative_l2=0.02,
        window_tokens=256,
        reference="CPU FP64; only checkpoint models round state, every 16 tokens",
    )

    def save():
        (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    save()
    tables = {k: [] for k in ["outputs", "states", "windows", "checks", "retention"]}
    try:
        csvwrite(out / "reference_checks.csv", validate())
        for name, pattern, t in cases:
            print(f"Checking {name}", flush=True)
            analyze(name, pattern, t, tables)
            for table, rows in tables.items():
                csvwrite(out / f"{table}.csv", rows)
        report(out)
        meta["status"] = "complete"
        save()
        print(out, flush=True)
    except BaseException as error:
        meta.update(status="failed", error=str(error))
        save()
        raise


if __name__ == "__main__":
    main()
