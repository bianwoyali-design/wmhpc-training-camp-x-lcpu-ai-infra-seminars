"""Discussion 2: CHUNK=16 instruction replacement, one reproducible entry point."""

import argparse
import csv
import ctypes as ct
import hashlib
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
SOURCE = ROOT / "csrc/compare.cu"
SHAPES = [
    ("gram", 16, 16, 128),
    ("projection", 16, 128, 128),
    ("local_mix", 16, 128, 16),
    ("state_update", 128, 128, 16),
    ("square", 16, 16, 16),
]
METHODS = [
    "sm80_w1",
    "sm80_w4",
    "sm80_w8",
    "sm80_pad_w4",
    "sm80_transpose_w4",
    "tc_direct",
    "tc_transpose",
]
FLAGS = ["-O2", "-std=c++17", "-gencode", "arch=compute_103a,code=sm_103a", "-lineinfo"]


def write_csv(path, rows):
    with Path(path).open("w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def hashes():
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [SOURCE, ROOT / "run.py"]
    }


def build():
    BUILD.mkdir(exist_ok=True)
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError("需要 CUDA 13 nvcc")
    stamp = dict(
        source=hashes()["csrc/compare.cu"],
        flags=FLAGS,
        nvcc=subprocess.check_output([nvcc, "--version"], text=True),
    )
    path = BUILD / "build.json"
    if (
        not path.exists()
        or json.loads(path.read_text()) != stamp
        or not (BUILD / "compare.so").exists()
    ):
        with (BUILD / "build.log").open("w") as log:
            subprocess.run(
                [
                    nvcc,
                    *FLAGS,
                    "--shared",
                    "-Xcompiler",
                    "-fPIC",
                    "--ptxas-options=-v",
                    str(SOURCE),
                    "-o",
                    str(BUILD / "compare.so"),
                ],
                stdout=log,
                stderr=log,
                check=True,
            )
        path.write_text(json.dumps(stamp, indent=2) + "\n")
    return stamp


def inspect():
    build()
    with (BUILD / "inspect.log").open("w") as log:
        for mode in ["ptx", "cubin"]:
            subprocess.run(
                [
                    "nvcc",
                    *FLAGS,
                    "-" + mode,
                    str(SOURCE),
                    "-o",
                    str(BUILD / f"compare.{mode}"),
                ],
                stdout=log,
                stderr=log,
                check=True,
            )
        with (BUILD / "compare.sass").open("w") as out:
            subprocess.run(
                ["nvdisasm", "-c", str(BUILD / "compare.cubin")],
                stdout=out,
                stderr=log,
                check=True,
            )
    ptx = (BUILD / "compare.ptx").read_text()
    sass = (BUILD / "compare.sass").read_text()
    opcodes = sorted(
        set(re.findall(r"\b(?:HMMA|UTCMMA|UTCHMMA|HGMMA)[.A-Z0-9_]*", sass))
    )
    if (
        "tcgen05.mma.cta_group::1.kind::f16" not in ptx
        or not any(x.startswith("HMMA") for x in opcodes)
        or not any(x.startswith("UTC") for x in opcodes)
    ):
        raise RuntimeError(f"缺少预期的两类指令：{opcodes}")
    return dict(
        target="sm_103a",
        opcodes=opcodes,
        source_sha256=hashes()["csrc/compare.cu"],
        ptx_tcgen05=True,
        note="Static instructions, not dynamic counts or a throughput measurement.",
    )


class Kernel:
    def __init__(self):
        self.lib = ct.CDLL(str(BUILD / "compare.so"))
        self.fn = self.lib.launch
        self.fn.argtypes = (
            [ct.c_void_p] * 3 + [ct.c_int] * 4 + [ct.c_void_p, ct.POINTER(ct.c_int)]
        )
        self.fn.restype = ct.c_int

    def run(self, a, b, o, shape, method, repeats):
        err = self.fn(
            a.data_ptr(),
            b.data_ptr(),
            o.data_ptr(),
            a.shape[0],
            shape,
            method,
            repeats,
            torch.cuda.current_stream().cuda_stream,
            None,
        )
        if err:
            raise RuntimeError(f"CUDA launch error {err}")

    def resources(self, shape, method):
        out = (ct.c_int * 4)()
        err = self.fn(None, None, None, 1, shape, method, 1, None, out)
        if err:
            raise RuntimeError(f"CUDA resource error {err}")
        return list(out)


def configure():
    global torch
    import torch

    torch.set_num_threads(1)
    if torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("需要 B300 sm_103a")
    torch.backends.cuda.matmul.allow_tf32 = False
    p = torch.cuda.get_device_properties(0)
    uuid = str(p.uuid)
    if not uuid.startswith("GPU-"):
        uuid = "GPU-" + uuid
    return dict(
        device=p.name,
        uuid=uuid,
        sm_count=p.multi_processor_count,
        torch=torch.__version__,
        cuda=torch.version.cuda,
    )


def accuracy(kernel, quick):
    rows = []
    for seed in [0] if quick else [0, 1, 2]:
        torch.manual_seed(seed)
        for shape, (name, m, n, k) in enumerate(SHAPES):
            for case in ["random", "integer", "zero"]:
                batch = 3 if quick else 8
                if case == "random":
                    a = torch.randn(batch, m, k, device="cuda").bfloat16()
                    b = torch.randn(batch, k, n, device="cuda").bfloat16()
                else:
                    a = torch.randint(-3, 4, (batch, m, k), device="cuda").bfloat16()
                    b = torch.randint(-3, 4, (batch, k, n), device="cuda").bfloat16()
                    if case == "zero":
                        a.zero_()
                ref = a.double() @ b.double()
                for repeats in [1, 32]:
                    for method, label in enumerate(METHODS):
                        out = torch.full((batch, m, n), float("nan"), device="cuda")
                        kernel.run(a, b, out, shape, method, repeats)
                        torch.cuda.synchronize()
                        gold = ref * repeats
                        diff = out.double() - gold
                        rel = (
                            (
                                torch.linalg.vector_norm(diff.flatten(1), dim=1)
                                / torch.linalg.vector_norm(
                                    gold.flatten(1), dim=1
                                ).clamp_min(1e-30)
                            )
                            .max()
                            .item()
                        )
                        exact = torch.equal(out.double(), gold)
                        if (
                            not torch.isfinite(out).all()
                            or rel > 2e-5
                            or (case != "random" and not exact)
                        ):
                            raise AssertionError(
                                (
                                    name,
                                    label,
                                    seed,
                                    case,
                                    repeats,
                                    rel,
                                    diff.abs().max().item(),
                                )
                            )
                        rows.append(
                            dict(
                                shape=name,
                                method=label,
                                seed=seed,
                                case=case,
                                batch=batch,
                                repeats=repeats,
                                relative_error=rel,
                                max_abs_error=diff.abs().max().item(),
                                exact=exact,
                            )
                        )
    return rows


def timing(kernel, args, out, device):
    rows = []
    resources = []
    rounds = 2 if args.quick else 5
    samples = 2 if args.quick else 5
    launches = 10 if args.quick else 30
    workloads = [1, 12, 96] if args.quick else [1, 12, 96, 6144, 49152]
    rng = random.Random(42)
    with (out / "clocks.csv").open("w") as log:
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "-i",
                device["uuid"],
                "--query-gpu=timestamp,uuid,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu",
                "--format=csv",
                "-lms",
                "200",
            ],
            stdout=log,
            stderr=subprocess.DEVNULL,
        )
        try:
            torch.manual_seed(777)
            for shape, (name, m, n, k) in enumerate(SHAPES):
                for method, label in enumerate(METHODS):
                    regs, smem, local, blocks = kernel.resources(shape, method)
                    resources.append(
                        dict(
                            shape=name,
                            method=label,
                            registers=regs,
                            shared_bytes=smem,
                            local_bytes=local,
                            max_ctas_per_sm=blocks,
                        )
                    )
                for batch in workloads:
                    a = torch.randn(batch, m, k, device="cuda").bfloat16()
                    b = torch.randn(batch, k, n, device="cuda").bfloat16()
                    o = torch.empty(batch, m, n, device="cuda")
                    for repeats in [1, 32] if batch in [1, 12, 96] else [1]:
                        graphs = []
                        for method in range(len(METHODS)):
                            for _ in range(5):
                                kernel.run(a, b, o, shape, method, repeats)
                            torch.cuda.synchronize()
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                for _ in range(launches):
                                    kernel.run(a, b, o, shape, method, repeats)
                            graphs.append(graph)
                        for round_id in range(rounds):
                            order = list(range(len(METHODS)))
                            rng.shuffle(order)
                            for method in order:
                                graph = graphs[method]
                                graph.replay()
                                torch.cuda.synchronize()
                                times = []
                                for _ in range(samples):
                                    start = torch.cuda.Event(enable_timing=True)
                                    end = torch.cuda.Event(enable_timing=True)
                                    start.record()
                                    graph.replay()
                                    end.record()
                                    end.synchronize()
                                    times.append(
                                        start.elapsed_time(end) * 1000 / launches
                                    )
                                rows.append(
                                    dict(
                                        shape=name,
                                        m=m,
                                        n=n,
                                        k=k,
                                        batch=batch,
                                        repeats=repeats,
                                        method=METHODS[method],
                                        round=round_id,
                                        median_us=statistics.median(times),
                                        min_us=min(times),
                                        max_us=max(times),
                                        us_per_gemm=statistics.median(times) / repeats,
                                    )
                                )
                        print(
                            f"Timed {name}, batch={batch}, repeats={repeats}",
                            flush=True,
                        )
                        del graphs, graph
                    del a, b, o
                    write_csv(out / "timing.csv", rows)
        finally:
            monitor.terminate()
            monitor.wait()
    write_csv(out / "resources.csv", resources)
    return rows


def report(out):
    with (out / "timing.csv").open() as f:
        rows = list(csv.DictReader(f))
    grouped = {}
    for row in rows:
        key = (row["shape"], int(row["batch"]), int(row["repeats"]), row["method"])
        grouped.setdefault(key, []).append(float(row["median_us"]))
    med = {k: statistics.median(v) for k, v in grouped.items()}
    lines = [
        "# CHUNK=16 指令替换测量",
        "",
        "单位 µs/kernel；每格为五轮中位数的中位数（quick 为两轮）。speedup=最佳 SM80 / 最佳 tcgen05。",
        "SM80 最佳候选包含原方向 1/4/8 warp 和转置 4 warp；补零 SM80 仅作控制组。",
        "",
        "| shape | batch | repeats | best SM80 | µs | tc direct µs | tc transpose µs | speedup |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for shape, batch, repeats in sorted({k[:3] for k in med}):
        base = (shape, batch, repeats)
        us, label = min(
            (med[(*base, label)], label) for label in METHODS[:3] + [METHODS[4]]
        )
        direct = med[(*base, "tc_direct")]
        trans = med[(*base, "tc_transpose")]
        lines.append(
            f"| {shape} | {batch} | {repeats} | {label} | {us:.4f} | {direct:.4f} | {trans:.4f} | {us / min(direct, trans):.3f}× |"
        )
    lines += [
        "",
        "repeats=32 是相同输入的 D += A×B，摊薄装载/分配/回写；不是 KDA 的状态递推。",
        "batch=12/96 是独立矩阵数；6144/49152 对应 8192/16 × 12/96 个独立 chunk-head，后者不是 K2 可用并行度。",
        "",
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "command", choices=["build", "inspect", "check", "reproduce", "report"]
    )
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if args.command == "report":
        if args.out is None:
            p.error("report 需要 --out 结果目录")
        report(args.out)
        return
    provenance = build()
    if args.command == "build":
        return
    if args.command == "inspect":
        print(json.dumps(inspect(), indent=2))
        return
    device = configure()
    kernel = Kernel()
    if args.command == "check":
        rows = accuracy(kernel, args.quick)
        print(f"PASS {len(rows)} accuracy rows")
        return
    out = (
        args.out
        or ROOT / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    ).resolve()
    out.mkdir(parents=True, exist_ok=False)
    meta = dict(
        status="running",
        **device,
        build=provenance,
        source_hashes=hashes(),
        quick=args.quick,
        slurm_job=os.getenv("SLURM_JOB_ID"),
        argv=sys.argv,
        rounds=2 if args.quick else 5,
        samples=2 if args.quick else 5,
        graph_launches=10 if args.quick else 30,
    )

    def save():
        (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    save()
    try:
        (out / "instructions.json").write_text(json.dumps(inspect(), indent=2) + "\n")
        shutil.copy2(BUILD / "build.log", out / "build.log")
        for tool in ["memcheck", "racecheck"]:
            cmd = [
                "compute-sanitizer",
                "--tool",
                tool,
                "--error-exitcode",
                "99",
                "--log-file",
                str(out / f"{tool}.log"),
                sys.executable,
                str(ROOT / "run.py"),
                "check",
                "--quick",
            ]
            subprocess.run(cmd, check=True)
            meta[tool] = "passed"
            save()
        rows = accuracy(kernel, args.quick)
        write_csv(out / "accuracy.csv", rows)
        meta["accuracy_rows"] = len(rows)
        save()
        timing(kernel, args, out, device)
        report(out)
        meta["status"] = "complete"
        save()
        print(f"Complete: {out}", flush=True)
    except BaseException as error:
        meta.update(status="failed", error=str(error))
        save()
        raise


if __name__ == "__main__":
    main()
