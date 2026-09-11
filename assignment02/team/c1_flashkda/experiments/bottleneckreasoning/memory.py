"""Resolve the L1TEX/L2/SM aggregate counters for two representative shapes."""

import argparse
import json
import os
import subprocess
import sys

from run import ROOT, digest, parse_raw, write_csv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=__import__("pathlib").Path, required=True)
    args = p.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    import flash_kda_C
    import torch

    meta = dict(
        status="running",
        job=os.getenv("SLURM_JOB_ID"),
        uuid=str(torch.cuda.get_device_properties(0).uuid),
        source_sha256=digest(__file__),
        target_sha256=digest(ROOT / "run.py"),
        extension_sha256=digest(flash_kda_C.__file__),
        commands=[],
    )

    def save():
        (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    save()
    rows = []
    try:
        for case in ["h96_n1", "h96_n8"]:
            cmd = [
                "ncu",
                "--target-processes",
                "all",
                "--profile-from-start",
                "off",
                "--kernel-name-base",
                "function",
                "--kernel-name",
                "regex:_flash_kda_fwd_(prepare|recurrence)",
                "--launch-count",
                "2",
                "--clock-control",
                "none",
                "--cache-control",
                "all",
                "--replay-mode",
                "kernel",
                "--section",
                "MemoryWorkloadAnalysis",
                "--metrics",
                ",".join(
                    [
                        "gpu__time_duration.sum",
                        "gpc__cycles_elapsed.avg.per_second",
                        "l1tex__data_pipe_lsu_wavefronts.sum",
                        "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
                        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
                        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
                        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
                        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
                    ]
                    + [
                        "breakdown:"
                        + unit
                        + "__throughput.avg.pct_of_peak_sustained_elapsed"
                        for unit in ["l1tex", "lts", "sm"]
                    ]
                ),
                "--export",
                str(out / case),
                sys.executable,
                str(ROOT / "run.py"),
                "target",
                "--case",
                case,
                "--out",
                str(out / f"{case}.check.json"),
            ]
            meta["commands"].append(cmd)
            save()
            print(f"Memory breakdown {case}", flush=True)
            with (out / f"{case}.log").open("w") as log:
                subprocess.run(cmd, stdout=log, stderr=log, check=True)
            raw = out / f"{case}.raw.csv"
            with raw.open("w") as f:
                subprocess.run(
                    [
                        "ncu",
                        "--import",
                        str(out / f"{case}.ncu-rep"),
                        "--page",
                        "raw",
                        "--csv",
                        "--print-units",
                        "base",
                    ],
                    stdout=f,
                    stderr=subprocess.PIPE,
                    check=True,
                )
            for r in parse_raw(raw):
                stage = "k1" if "_prepare" in r["Kernel Name"] else "k2"
                rows.append(
                    dict(
                        case=case,
                        stage=stage,
                        metric=r["Metric Name"],
                        unit=r["Metric Unit"],
                        value=r["Metric Value"],
                    )
                )
        write_csv(out / "metrics.csv", rows)
        meta["status"] = "complete"
        save()
        print(out)
    except BaseException as error:
        meta.update(status="failed", error=str(error))
        save()
        raise


if __name__ == "__main__":
    main()
