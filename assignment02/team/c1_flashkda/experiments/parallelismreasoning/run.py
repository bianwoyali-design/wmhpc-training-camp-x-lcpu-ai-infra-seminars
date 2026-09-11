"""Discussion 3: official K2 concurrency, scheduling models, and algebra counterexamples."""

import argparse
import collections
import csv
import ctypes as ct
import hashlib
import heapq
import json
import math
import os
import random
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FLASH = ROOT.parent.parent / "FlashKDA"
BUILD = ROOT / "build"
FIELDS = [
    "type",
    "gx",
    "gy",
    "gz",
    "bx",
    "by",
    "bz",
    "dynamic_shared",
    "registers",
    "static_shared",
    "local_bytes",
    "max_ctas_per_sm",
]


def write_csv(path, rows):
    with Path(path).open("w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build():
    BUILD.mkdir(exist_ok=True)
    source = ROOT / "csrc/graph.cpp"
    stamp = {
        "source": digest(source),
        "nvcc": subprocess.check_output(["nvcc", "--version"], text=True),
    }
    manifest = BUILD / "build.json"
    if (
        not (BUILD / "graph.so").exists()
        or not manifest.exists()
        or json.loads(manifest.read_text()) != stamp
    ):
        with (BUILD / "build.log").open("w") as log:
            subprocess.run(
                [
                    "nvcc",
                    "-O2",
                    "-std=c++17",
                    "--shared",
                    "-Xcompiler",
                    "-fPIC",
                    str(source),
                    "-lcuda",
                    "-o",
                    str(BUILD / "graph.so"),
                ],
                stdout=log,
                stderr=log,
                check=True,
            )
        manifest.write_text(json.dumps(stamp, indent=2) + "\n")
    return stamp


class GraphControl:
    def __init__(self, graph):
        self.graph = graph
        self.lib = ct.CDLL(str(BUILD / "graph.so"))
        self.lib.node_count.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int)]
        self.lib.describe.argtypes = [
            ct.c_void_p,
            ct.c_int,
            ct.POINTER(ct.c_void_p),
            ct.c_char_p,
            ct.c_int,
            ct.POINTER(ct.c_int),
        ]
        self.lib.enable.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int]
        self.nodes = []
        count = ct.c_int()
        self.check(self.lib.node_count(graph.raw_cuda_graph(), ct.byref(count)))
        for i in range(count.value):
            node = ct.c_void_p()
            name = ct.create_string_buffer(32768)
            info = (ct.c_int * 12)()
            self.check(
                self.lib.describe(
                    graph.raw_cuda_graph(), i, ct.byref(node), name, len(name), info
                )
            )
            raw = name.value.decode()
            stage = "other"
            if "_flash_kda_fwd_prepare" in raw:
                stage = "k1"
            elif "_flash_kda_fwd_recurrence" in raw:
                stage = "k2"
            elif "_flash_kda_build_tile_prefix" in raw:
                stage = "prefix"
            self.nodes.append((node, dict(zip(FIELDS, info)), stage, raw))
        assert any(s == "k1" for _, _, s, _ in self.nodes) and any(
            s == "k2" for _, _, s, _ in self.nodes
        )
        assert all(info["type"] == 0 for _, info, _, _ in self.nodes), (
            "Unexpected non-kernel graph node"
        )

    @staticmethod
    def check(error):
        if error:
            raise RuntimeError(f"CUDA graph API error {error}")

    def mode(self, mode):
        for node, info, stage, name in self.nodes:
            self.check(
                self.lib.enable(
                    self.graph.raw_cuda_graph_exec(),
                    node,
                    int(mode == "full" or stage == mode),
                )
            )


def cases(quick):
    heads = [12, 96] if quick else [12, 24, 48, 96]
    for h in heads:
        for n in [1, 8] if quick else [1, 2, 4, 8, 16]:
            yield f"equal_n{n}", h, [8192 // n] * n, False
        yield "skew8", h, [4608] + [512] * 7, True
        if not quick:
            yield "ragged6", h, [1300, 547, 2048, 963, 271, 3063], True
            yield "equal8_varlen", h, [1024] * 8, True
    if not quick:
        for t in [128, 512, 2048]:
            yield f"length{t}", 12, [t], False


def schedule_models(case_rows, sms):
    rows = []
    for name, h, lengths, varlen in case_rows:
        chunks = [math.ceil(n / 16) for n in lengths]
        # grid.x(sequence) is the fastest varying index in the official launch.
        tasks = chunks * h
        for slots in [sms, 2 * sms]:

            def queue(order):
                workers = [0] * slots
                heapq.heapify(workers)
                for cost in order:
                    heapq.heappush(workers, heapq.heappop(workers) + cost)
                return max(workers)

            static = [0] * slots
            for i, cost in enumerate(tasks):
                static[i % slots] += cost
            rows.append(
                dict(
                    case=name,
                    heads=h,
                    sequences=len(lengths),
                    tasks=len(tasks),
                    slots=slots,
                    total_chunk_heads=sum(tasks),
                    critical_path_chunks=max(tasks),
                    lower_bound=max(max(tasks), sum(tasks) / slots),
                    fifo=queue(tasks),
                    longest_first=queue(sorted(tasks, reverse=True)),
                    static_round_robin=max(static),
                )
            )
    return rows


def prepare_case(h, lengths, varlen, launches):
    import flash_kda_C as extension
    import torch

    n = len(lengths)
    total = sum(lengths)
    t = total if varlen else lengths[0]
    b = 1 if varlen else n
    torch.manual_seed(100 + h + n)
    shape = (b, t, h, 128)
    q = torch.nn.functional.normalize(
        torch.randn(shape, device="cuda"), dim=-1
    ).bfloat16()
    k = torch.nn.functional.normalize(
        torch.randn(shape, device="cuda"), dim=-1
    ).bfloat16()
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    g = torch.randn_like(v)
    beta = torch.randn(b, t, h, device="cuda", dtype=torch.bfloat16)
    alog = torch.rand(h, device="cuda")
    bias = torch.rand(h, 128, device="cuda")
    initial = (0.1 * torch.randn(n, h, 128, 128, device="cuda")).bfloat16()
    final = torch.empty_like(initial)
    out = torch.empty_like(v)
    workspace = torch.empty(
        extension.get_workspace_size(total, h, n), device="cuda", dtype=torch.uint8
    )
    cu = (
        torch.tensor(
            [0] + list(__import__("itertools").accumulate(lengths)),
            device="cuda",
            dtype=torch.int64,
        )
        if varlen
        else None
    )

    def invoke():
        extension.fwd(
            q,
            k,
            v,
            g,
            beta,
            128**-0.5,
            out,
            workspace,
            alog,
            bias,
            -5.0,
            initial,
            final,
            cu,
        )

    for _ in range(3):
        invoke()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        for _ in range(launches):
            invoke()
    graph.instantiate()
    control = GraphControl(graph)
    graph.replay()
    torch.cuda.synchronize()
    gold = out.clone()
    sref = final.clone()
    assert torch.isfinite(gold).all() and torch.isfinite(sref).all()
    # Prove stage isolation preserves all output/state, not merely that the graph launches.
    control.mode("k2")
    out.fill_(float("nan"))
    final.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, gold) and torch.equal(final, sref), (
        "K2 isolated replay differs"
    )
    control.mode("k1")
    graph.replay()
    torch.cuda.synchronize()
    control.mode("k2")
    out.fill_(float("nan"))
    final.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, gold) and torch.equal(final, sref), (
        "K1/K2 isolated replay differs"
    )
    keep = (
        q,
        k,
        v,
        g,
        beta,
        alog,
        bias,
        initial,
        final,
        out,
        workspace,
        cu,
        gold,
        sref,
    )
    return graph, control, keep


def benchmark(args, out):
    import flash_kda_C
    import torch

    torch.set_num_threads(1)
    if torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("需要 B300")
    dev = torch.cuda.get_device_properties(0)
    uuid = str(dev.uuid)
    if not uuid.startswith("GPU-"):
        uuid = "GPU-" + uuid
    spec = list(cases(args.quick))
    launches = 3 if args.quick else 10
    rounds = 2 if args.quick else 5
    samples = 2 if args.quick else 5
    meta = dict(
        status="running",
        device=dev.name,
        uuid=uuid,
        sm_count=dev.multi_processor_count,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        slurm_job=os.getenv("SLURM_JOB_ID"),
        quick=args.quick,
        graph_launches=launches,
        rounds=rounds,
        samples=samples,
        extension=str(flash_kda_C.__file__),
        extension_sha256=digest(Path(flash_kda_C.__file__)),
        source_hashes={
            str(p.relative_to(ROOT)): digest(p)
            for p in [ROOT / "run.py", ROOT / "proofs.py", ROOT / "csrc/graph.cpp"]
        },
        official_hashes={
            str(p.relative_to(FLASH)): digest(p)
            for p in [
                FLASH / "csrc/flash_kda.cpp",
                FLASH / "csrc/smxx/fwd_launch.cu",
                FLASH / "csrc/smxx/fwd_kernel2.cuh",
            ]
        },
    )

    def save():
        (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    save()
    rows = []
    nodes = []
    checks = []
    rng = random.Random(42)
    try:
        from proofs import validate

        proof_rows = validate()
        write_csv(out / "algebra.csv", proof_rows)
        write_csv(
            out / "scheduling_model.csv",
            schedule_models(spec, dev.multi_processor_count),
        )
        with (out / "clocks.csv").open("w") as log:
            monitor = subprocess.Popen(
                [
                    "nvidia-smi",
                    "-i",
                    uuid,
                    "--query-gpu=timestamp,uuid,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu",
                    "--format=csv",
                    "-lms",
                    "200",
                ],
                stdout=log,
                stderr=subprocess.DEVNULL,
            )
            try:
                for name, h, lengths, varlen in spec:
                    graph, control, keep = prepare_case(h, lengths, varlen, launches)
                    count = collections.Counter(
                        stage for _, _, stage, _ in control.nodes
                    )
                    assert count["k1"] == count["k2"] == launches
                    for stage in count:
                        entry = next(
                            (info, raw)
                            for _, info, s, raw in control.nodes
                            if s == stage
                        )
                        nodes.append(
                            dict(
                                case=name,
                                heads=h,
                                stage=stage,
                                count=count[stage],
                                **entry[0],
                                kernel=entry[1],
                            )
                        )
                    checks.append(
                        dict(
                            case=name,
                            heads=h,
                            sequences=len(lengths),
                            seq_lens=json.dumps(lengths),
                            isolated_output_exact=True,
                            isolated_state_exact=True,
                        )
                    )
                    for rid in range(rounds):
                        modes = ["full", "k1", "k2"]
                        rng.shuffle(modes)
                        for mode in modes:
                            control.mode(mode)
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
                                times.append(start.elapsed_time(end) * 1000 / launches)
                            rows.append(
                                dict(
                                    case=name,
                                    heads=h,
                                    sequences=len(lengths),
                                    tokens=sum(lengths),
                                    max_chunks=max(math.ceil(x / 16) for x in lengths),
                                    chunk_heads=sum(math.ceil(x / 16) for x in lengths)
                                    * h,
                                    k2_ctas=len(lengths) * h,
                                    mode=mode,
                                    round=rid,
                                    median_us=statistics.median(times),
                                    min_us=min(times),
                                    max_us=max(times),
                                )
                            )
                    write_csv(out / "timing.csv", rows)
                    write_csv(out / "kernels.csv", nodes)
                    write_csv(out / "checks.csv", checks)
                    del control, graph, keep
                    print(f"Completed {name} H={h}", flush=True)
            finally:
                monitor.terminate()
                monitor.wait()
        meta.update(status="complete", cases=len(spec), algebra_rows=len(proof_rows))
        save()
        report(out)
    except BaseException as error:
        meta.update(status="failed", error=str(error))
        save()
        raise


def report(out):
    with (out / "timing.csv").open() as f:
        rows = list(csv.DictReader(f))
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["case"], int(r["heads"]), r["mode"])].append(float(r["median_us"]))
    lines = [
        "# 官方 FlashKDA 并行度测量",
        "",
        "单位 µs/forward 或 µs/stage；各轮中位数再取中位数。K1/K2 独立计时使用预先生成的同一输入。",
        "",
        "| case | H | K2 CTA 数 | full µs | K1 µs | K2 µs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, h in sorted({k[:2] for k in groups}):
        row = next(r for r in rows if r["case"] == name and int(r["heads"]) == h)
        lines.append(
            f"| {name} | {h} | {row['k2_ctas']} | "
            + " | ".join(
                f"{statistics.median(groups[(name, h, m)]):.3f}"
                for m in ["full", "k1", "k2"]
            )
            + " |"
        )
    lines += [
        "",
        "不同序列划分是不同的独立序列工作负载，不是把同一条序列任意切断后的等价加速。",
        "full 还包括 beta 转置、varlen prefix；独立计时改变缓存/调度，不能严格相加还原 full。",
        "",
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "check", "reproduce", "report"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "report":
        if args.out is None:
            parser.error("report requires --out")
        report(args.out)
        return
    if args.command == "check":
        from proofs import validate

        print(json.dumps(validate(), indent=2))
        return
    build()
    if args.command == "build":
        return
    out = (
        args.out
        or ROOT / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    ).resolve()
    out.mkdir(parents=True, exist_ok=False)
    benchmark(args, out)
    print(out)


if __name__ == "__main__":
    main()
