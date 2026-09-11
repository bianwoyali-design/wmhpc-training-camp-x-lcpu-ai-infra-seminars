"""Shared inputs, CUDA bindings, checks and timing for discussion point 1."""

import csv
import ctypes as ct
import math
import random
import statistics
import subprocess
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch

CHUNKS = (16, 32, 64)
DECAY_METHODS = ("raw", "center", "tile16", "direct")
STAGES = ("gram", "projection", "local_mix", "state_update")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def shape(stage, c):
    return ((c, c, 128), (c, 128, 128), (c, 128, c), (128, 128, c))[stage]


def methods(c, warps):
    return [f"unified_w{w}" for w in warps] + (["official"] if c == 16 else [])


def doubling(l):
    x = torch.eye(l.shape[-1], device=l.device, dtype=l.dtype) - l
    p = l
    for _ in range(int(math.log2(l.shape[-1])) - 1):
        p = p @ p
        x = x + x @ p
    return x.bfloat16().float()


class Kernels:
    def __init__(self, library):
        self.lib = ct.CDLL(str(library))
        signatures = {
            "launch_official": [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_void_p],
            "launch_config": [
                ct.c_void_p,
                ct.c_void_p,
                ct.c_int,
                ct.c_int,
                ct.c_int,
                ct.c_void_p,
            ],
            "resources_config": [ct.c_int, ct.c_int, ct.POINTER(ct.c_int)],
            "launch_decay": [
                ct.c_void_p,
                ct.c_void_p,
                ct.c_int,
                ct.c_int,
                ct.c_int,
                ct.c_void_p,
            ],
            "launch_mma": [
                ct.c_void_p,
                ct.c_void_p,
                ct.c_void_p,
                ct.c_int,
                ct.c_int,
                ct.c_int,
                ct.c_int,
                ct.c_void_p,
            ],
        }
        for name, sig in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = sig, ct.c_int

    def invoke(self, name, *args):
        error = getattr(self.lib, name)(*args)
        if error:
            raise RuntimeError(f"{name}: CUDA error {error}")

    def inverse(self, l, out, method):
        params = [l.data_ptr(), out.data_ptr(), l.shape[0]]
        fn = "launch_official"
        if method.startswith("unified_w"):
            fn = "launch_config"
            params += [l.shape[-1], int(method.split("_w")[1])]
        self.invoke(fn, *params, torch.cuda.current_stream().cuda_stream)

    def decay(self, gates, out, mode):
        self.invoke(
            "launch_decay",
            gates.data_ptr(),
            out.data_ptr(),
            gates.shape[0],
            gates.shape[1],
            mode,
            torch.cuda.current_stream().cuda_stream,
        )

    def mma(self, a, b, out, c, stage, warps):
        self.invoke(
            "launch_mma",
            a.data_ptr(),
            b.data_ptr(),
            out.data_ptr(),
            a.shape[0],
            c,
            stage,
            warps,
            torch.cuda.current_stream().cuda_stream,
        )

    def resources(self, warps):
        rows = []
        for c in CHUNKS:
            for w in warps:
                a = (ct.c_int * 4)()
                self.invoke("resources_config", c, w, a)
                rows.append(
                    dict(
                        chunk=c,
                        warps=w,
                        registers_per_thread=a[0],
                        shared_bytes=a[1],
                        local_bytes_per_thread=a[2],
                        max_blocks_per_sm=a[3],
                    )
                )
        return rows


def configure():
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("需要在 Slurm GPU 分配内运行；登录节点只能 build/report。")
    if torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("本实验编译目标为 B300 sm_103a。")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    p = torch.cuda.get_device_properties(0)
    uuid = str(getattr(p, "uuid", "unknown"))
    if uuid != "unknown" and not uuid.startswith("GPU-"):
        uuid = "GPU-" + uuid
    return dict(
        device=p.name,
        device_uuid=uuid,
        multiprocessors=p.multi_processor_count,
        torch=torch.__version__,
        cuda=torch.version.cuda,
    )


def inverse_checks(kernels, settings):
    rows = []
    for seed in settings["seeds"]:
        torch.manual_seed(seed)
        n = settings["accuracy_batch"]
        k = torch.nn.functional.normalize(
            torch.randn(n, 64, 128, device="cuda"), dim=-1
        )
        s = (-0.1 * torch.rand_like(k)).cumsum(1)
        beta = torch.sigmoid(torch.randn(n, 64, device="cuda"))
        cases = {
            "structured": torch.tril(
                ((k * s.exp()) @ (k * (-s).exp()).transpose(-1, -2)) * beta[:, :, None],
                -1,
            ),
            "zero": torch.zeros(n, 64, 64, device="cuda"),
            "stress": torch.tril(0.2 * torch.randn(n, 64, 64, device="cuda"), -1),
        }
        for case, full in cases.items():
            for c in CHUNKS:
                l = full[:, :c, :c].half().float().contiguous()
                eye = torch.eye(c, device="cuda", dtype=torch.float64).expand_as(l)
                a = eye + l.double()
                ref = torch.linalg.solve_triangular(
                    a, eye, upper=False, unitriangular=True
                )
                model = doubling(l.half()).double()
                warp_ref = None
                for method in methods(c, settings["warps"]):
                    x = torch.empty_like(l)
                    kernels.inverse(l, x, method)
                    torch.cuda.synchronize()
                    x = x.double()
                    if not torch.isfinite(x).all():
                        raise AssertionError((case, c, method, "nonfinite inverse"))
                    if method.startswith("unified"):
                        if warp_ref is None:
                            warp_ref = x.clone()
                        elif not torch.equal(warp_ref, x):
                            raise AssertionError(
                                (case, c, method, "warp outputs differ")
                            )
                    res = (
                        (torch.linalg.matrix_norm(a @ x - eye) / math.sqrt(c))
                        .max()
                        .item()
                    )
                    if case == "zero" and not torch.equal(x, eye):
                        raise AssertionError("L=0 should produce exact identity")
                    if case == "structured" and res >= 0.02:
                        raise AssertionError((c, method, res))
                    rows.append(
                        dict(
                            seed=seed,
                            case=case,
                            chunk=c,
                            method=method,
                            max_residual=res,
                            max_relative_error=(
                                torch.linalg.matrix_norm(x - ref)
                                / torch.linalg.matrix_norm(ref)
                            )
                            .max()
                            .item(),
                            max_abs_difference_from_torch_fp16=(x - model)
                            .abs()
                            .max()
                            .item(),
                        )
                    )
    return rows


def gate_cases(seed, n):
    gen = torch.Generator().manual_seed(seed)
    cases = {
        f"constant_{g:g}": torch.full((n, 64), g) for g in (0.0, -0.1, -1.0, -3.0, -5.0)
    }
    cases["uniform"] = -5 * torch.rand(n, 64, generator=gen)
    cases["mixed"] = torch.where(torch.rand(n, 64, generator=gen) < 0.1, -5.0, -0.01)
    return cases


def range_checks(kernels, settings):
    rows = []
    for seed in settings["seeds"]:
        for name, full in gate_cases(seed, settings["accuracy_batch"]).items():
            if seed != settings["seeds"][0] and name.startswith("constant"):
                continue
            for c in CHUNKS:
                g = full[:, :c].contiguous().cuda()
                s = g.double().cumsum(1)
                mask = torch.ones(c, c, device="cuda", dtype=torch.bool).tril()
                ref = torch.exp(s[:, :, None] - s[:, None, :])[:, mask]
                for mode, method in enumerate(DECAY_METHODS):
                    result = torch.empty(g.shape[0], c, c, device="cuda")
                    kernels.decay(g, result, mode)
                    torch.cuda.synchronize()
                    x = result.double()[:, mask]
                    finite = bool(torch.isfinite(x).all())
                    rel = (
                        (
                            torch.linalg.vector_norm(x - ref, dim=1)
                            / torch.linalg.vector_norm(ref, dim=1)
                        )
                        .max()
                        .item()
                        if finite
                        else None
                    )
                    if method in ("tile16", "direct") and (not finite or rel >= 0.02):
                        raise AssertionError((name, c, method, rel))
                    if name == "constant_0" and not torch.equal(x, ref):
                        raise AssertionError("zero gate mismatch")
                    if (
                        name == "constant_-5"
                        and method == "center"
                        and c == 32
                        and not finite
                    ):
                        raise AssertionError("centered C=32 should be finite")
                    rows.append(
                        dict(
                            seed=seed,
                            case=name,
                            chunk=c,
                            method=method,
                            elements=x.numel(),
                            zero_count=int((x == 0).sum()),
                            inf_count=int(torch.isinf(x).sum()),
                            nan_count=int(torch.isnan(x).sum()),
                            max_relative_error=rel,
                            max_abs_error=(x - ref).abs().max().item()
                            if finite
                            else None,
                        )
                    )
    return rows


def mma_checks(kernels, settings):
    rows = []
    for seed in settings["seeds"]:
        torch.manual_seed(seed)
        for c in CHUNKS:
            for stage, name in enumerate(STAGES):
                m, n, k = shape(stage, c)
                a = torch.randn(
                    settings["accuracy_batch"], m, k, device="cuda"
                ).bfloat16()
                b = torch.randn(
                    settings["accuracy_batch"], k, n, device="cuda"
                ).bfloat16()
                ref = a.double() @ b.double()
                warp_ref = None
                for w in settings["warps"]:
                    x = torch.empty(a.shape[0], m, n, device="cuda")
                    kernels.mma(a, b, x, c, stage, w)
                    torch.cuda.synchronize()
                    if warp_ref is None:
                        warp_ref = x.clone()
                    elif not torch.equal(warp_ref, x):
                        raise AssertionError((name, c, w, "MMA warp mismatch"))
                    rel = (
                        (
                            torch.linalg.matrix_norm(x.double() - ref)
                            / torch.linalg.matrix_norm(ref)
                        )
                        .max()
                        .item()
                    )
                    if not torch.isfinite(x).all() or rel > 1e-5:
                        raise AssertionError((name, c, w, rel))
                    rows.append(
                        dict(
                            seed=seed,
                            stage=name,
                            chunk=c,
                            warps=w,
                            m=m,
                            n=n,
                            k=k,
                            max_relative_error=rel,
                            max_abs_error=(x.double() - ref).abs().max().item(),
                        )
                    )
    return rows


def checks(kernels, settings, suites):
    functions = {"inverse": inverse_checks, "range": range_checks, "mma": mma_checks}
    return {
        f"{suite}_accuracy": functions[suite](kernels, settings) for suite in suites
    }


@dataclass
class TimingCase:
    info: dict
    run: object


def timing_cases(kernels, settings, suite):
    # Tensors referenced by partial() stay alive until this suite finishes.
    cases = []
    for tokens in settings["tokens"]:
        workload = "single" if tokens is None else f"tokens_{tokens}"
        for c in CHUNKS:
            batch = 1 if tokens is None else tokens // c
            common = dict(workload=workload, chunk=c, batch=batch)
            torch.manual_seed(123)
            if suite == "inverse":
                l = (
                    torch.tril(0.02 * torch.randn(batch, c, c, device="cuda"), -1)
                    .half()
                    .float()
                    .contiguous()
                )
                out = torch.empty_like(l)
                for method in methods(c, settings["warps"]):
                    cases.append(
                        TimingCase(
                            dict(**common, method=method),
                            partial(kernels.inverse, l, out, method),
                        )
                    )
            elif suite == "range":
                # Mild gates keep all baselines numerically valid during timing.
                gates = -0.1 * torch.rand(batch, c, device="cuda")
                out = torch.empty(batch, c, c, device="cuda")
                for mode, method in enumerate(DECAY_METHODS):
                    cases.append(
                        TimingCase(
                            dict(**common, method=method),
                            partial(kernels.decay, gates, out, mode),
                        )
                    )
            else:
                for stage, name in enumerate(STAGES):
                    m, n, k = shape(stage, c)
                    a = torch.randn(batch, m, k, device="cuda").bfloat16()
                    b = torch.randn(batch, k, n, device="cuda").bfloat16()
                    out = torch.empty(batch, m, n, device="cuda")
                    for w in settings["warps"]:
                        info = dict(
                            **common,
                            stage=name,
                            method=f"wmma_w{w}",
                            m=m,
                            n=n,
                            k=k,
                            wmma_per_matrix=(m // 16) * (n // 16) * (k // 16),
                            padding_fraction=0,
                        )
                        cases.append(
                            TimingCase(
                                info, partial(kernels.mma, a, b, out, c, stage, w)
                            )
                        )
    return cases


def benchmark(kernels, settings, suite, output, device_uuid):
    cases = timing_cases(kernels, settings, suite)
    captured = []
    for case in cases:
        for _ in range(10):
            case.run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(settings["graph_launches"]):
                case.run()
        captured.append((case, graph))
    log = (output / f"{suite}_clocks.csv").open("w")
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,uuid,clocks.sm,power.draw,temperature.gpu",
        "--format=csv",
        "-lms",
        "200",
    ]
    if device_uuid.startswith("GPU-"):
        command += ["-i", device_uuid]
    monitor = None
    rows = []
    try:
        monitor = subprocess.Popen(command, stdout=log, stderr=subprocess.DEVNULL)
        rng = random.Random(77)
        for round_id in range(settings["rounds"]):
            order = list(captured)
            rng.shuffle(order)
            for case, graph in order:
                graph.replay()
                torch.cuda.synchronize()
                samples = []
                for _ in range(settings["samples"]):
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    samples.append(
                        start.elapsed_time(end) * 1000 / settings["graph_launches"]
                    )
                us = statistics.median(samples)
                rows.append(
                    dict(
                        round=round_id,
                        **case.info,
                        median_us=us,
                        min_us=min(samples),
                        max_us=max(samples),
                        ns_per_token=us
                        * 1000
                        / (case.info["batch"] * case.info["chunk"]),
                    )
                )
    finally:
        if monitor is not None:
            monitor.terminate()
            monitor.wait(timeout=10)
        log.close()
    return rows
