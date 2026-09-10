"""CHUNK=16 reasoning: validate, build, check, bench, reproduce, report. Run from any directory."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FLASH = ROOT.parent.parent / "FlashKDA"
BUILD = ROOT / "build"
LIBRARY = BUILD / "kernels.so"
SUITES = ("range", "inverse", "mma")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(force=False):
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError("找不到 nvcc；请加载 CUDA 13 工具链。")
    for required in [FLASH/"cutlass/include/cute/tensor.hpp", FLASH/"csrc/smxx/utils.cuh"]:
        if not required.exists():
            raise RuntimeError(f"缺少构建依赖：{required}；见 README 的环境要求。")
    sources = [ROOT/"csrc"/name for name in ("official.cu", "unified.cu", "decay.cu", "mma.cu")]
    dependencies = sources + [FLASH/"csrc/smxx/utils.cuh", FLASH/"cutlass/include/cutlass/bfloat16.h", FLASH/"cutlass/include/cute/arch/mma_sm80.hpp"]
    flags = ["-O2", "-std=c++17", "--extended-lambda", "-arch=sm_103a", "--shared", "-Xcompiler", "-fPIC", "--ptxas-options=-v"]
    provenance = dict(nvcc=subprocess.check_output([nvcc,"--version"],text=True), flags=flags,
                      hashes={str(f.relative_to(ROOT)) if f.is_relative_to(ROOT) else str(f.relative_to(FLASH)):digest(f) for f in dependencies})
    commit = subprocess.run(["git","-C",str(FLASH/"cutlass"),"rev-parse","HEAD"],capture_output=True,text=True)
    provenance["cutlass_commit"] = commit.stdout.strip() if commit.returncode==0 else "unknown"
    BUILD.mkdir(exist_ok=True)
    stamp=BUILD/"manifest.json"
    if not force and LIBRARY.exists() and stamp.exists() and json.loads(stamp.read_text())==provenance:
        print("Build cache: kernels.so",flush=True)
        return provenance
    command = [nvcc,*flags,"-I",str(FLASH/"cutlass/include"),"-I",str(FLASH/"csrc/smxx"),*[str(s) for s in sources],"-o",str(LIBRARY)]
    print("Building CUDA kernels ...",flush=True)
    with (BUILD/"build.log").open("w") as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"编译失败；查看 {BUILD/'build.log'}\n"+(BUILD/"build.log").read_text()[-4000:])
    stamp.write_text(json.dumps(provenance,indent=2)+"\n")
    return provenance


def inspect_instructions(destination):
    """Check actual emitted BF16 warp MMA instructions; static evidence only."""
    nvcc, disasm = shutil.which("nvcc"), shutil.which("nvdisasm")
    if not nvcc or not disasm:
        raise RuntimeError("指令检查需要 nvcc 与 nvdisasm。")
    source = ROOT/"csrc/mma.cu"
    base = [nvcc, "-O2", "-std=c++17", "-arch=sm_103a"]
    with (BUILD/"inspect.log").open("w") as log:
        subprocess.run([*base,"-ptx",str(source),"-o",str(BUILD/"mma.ptx")],stdout=log,stderr=log,check=True)
        subprocess.run([*base,"-cubin",str(source),"-o",str(BUILD/"mma.cubin")],stdout=log,stderr=log,check=True)
        with (BUILD/"mma.sass").open("w") as sass:
            subprocess.run([disasm,"-c",str(BUILD/"mma.cubin")],stdout=sass,stderr=log,check=True)
    ptx=(BUILD/"mma.ptx").read_text();sass=(BUILD/"mma.sass").read_text()
    opcodes=sorted(set(re.findall(r"\b(?:HMMA|HGMMA|UTCMMA)[.A-Z0-9]*",sass)))
    if "HMMA.16816.F32.BF16" not in opcodes:
        raise RuntimeError("没有找到预期的 BF16 HMMA，检查 build/mma.sass。")
    evidence=dict(source_sha256=digest(source),target="sm_103a",
        ptx_mma=sorted(set(re.findall(r"wmma\.mma\.sync[.a-zA-Z0-9]*",ptx))),
        sass_opcodes=opcodes,static_hmma_count=sass.count("HMMA.16816.F32.BF16"),
        note="Static disassembly of the same source/target/optimization; count is not dynamic executed instruction count.")
    destination.write_text(json.dumps(evidence,indent=2)+"\n")
    return evidence


def validate_range(output):
    from numerics import compare
    output = Path(output).resolve()
    if output.exists():
        raise RuntimeError(f"结果目录已存在，不覆盖：{output}")
    output.mkdir(parents=True)
    BUILD.mkdir(exist_ok=True)
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError("找不到 nvcc。")
    binary = BUILD/"ftz"
    command = [nvcc, "-O2", "-arch=sm_103a", "--ftz=false", "--fmad=false",
               "-lineinfo", str(ROOT/"csrc/ftz.cu"), "-o", str(binary)]
    with (output/"build.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=log, check=True)
    compare(output, "prepare")
    with (output/"device.log").open("w") as log:
        subprocess.run([str(binary), str(output/"inputs.txt"), str(output)], stdout=log, stderr=log, check=True)
    compare(output, "compare")
    (output/"provenance.json").write_text(json.dumps(dict(
        command=command, nvcc=subprocess.check_output([nvcc,"--version"],text=True),
        source_hashes={name:digest(ROOT/name) for name in ("numerics.py","csrc/ftz.cu")}
    ), indent=2)+"\n")


def settings(args):
    return dict(warps=args.warps,seeds=[0] if args.quick else [0,1,2],accuracy_batch=4 if args.quick else 8,
                graph_launches=10 if args.quick else 100,rounds=2 if args.quick else 5,samples=2 if args.quick else 5,
                tokens=[None,8192] if args.quick else [None,8192,8192*96],quick=args.quick)


def parse_warps(value):
    try:
        values=sorted(set(map(int,value.split(','))))
    except ValueError as error:
        raise argparse.ArgumentTypeError("warps 应为 1,4,8 的非空子集") from error
    if not values or not set(values)<=set([1,4,8]):
        raise argparse.ArgumentTypeError("只支持 warps=1,4,8")
    return values


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    subs=p.add_subparsers(dest="command",required=True)
    b=subs.add_parser("build",help="在登录节点或 GPU 节点构建全部 kernel")
    b.add_argument("--force",action="store_true")
    v=subs.add_parser("validate",help="FTZ 指令与 CPU 数值模型对照")
    v.add_argument("--out",type=Path)
    subs.add_parser("inspect",help="编译并反汇编 MMA；不需要 GPU")
    for cmd in ("check","bench","reproduce"):
        q=subs.add_parser(cmd,help={"check":"只运行正确性检查", "bench":"正确性检查 + 计时 + 汇总", "reproduce":"构建 + racecheck + 正确性 + 计时 + 汇总"}[cmd])
        q.add_argument("--suite",choices=["all",*SUITES],default="all")
        q.add_argument("--warps",type=parse_warps,default=[1,4,8])
        q.add_argument("--quick",action="store_true",help="缩小检查/采样规模；仅用于冒烟测试")
        if cmd!="check":
            q.add_argument("--out",type=Path,help="新结果目录；必须不存在，防止覆盖历史记录")
        if cmd=="reproduce":
            q.add_argument("--skip-racecheck",action="store_true",help="显式跳过 sanitizer，metadata 会记录")
    r=subs.add_parser("report",help="从计时 CSV 重新生成汇总；无需 GPU")
    r.add_argument("directory",type=Path)
    r.add_argument("--output",type=Path,help="默认写入所给目录 SUMMARY.md")
    return p


def main():
    args=parser().parse_args()
    if args.command=="validate":
        validate_range(args.out or ROOT/"runs"/datetime.now(timezone.utc).strftime("ftz-%Y%m%dT%H%M%S.%fZ"))
        return
    if args.command=="report":
        from report import render
        destination=args.output or args.directory/"SUMMARY.md"
        destination.write_text(render(args.directory))
        print(destination.resolve())
        return
    if args.command=="inspect":
        build()
        print(json.dumps(inspect_instructions(BUILD/"instructions.json"),indent=2))
        return
    if args.command=="build":
        build(args.force)
        return
    provenance=build()
    from experiment import Kernels,configure,checks,benchmark,write_csv
    device=configure()
    kernels=Kernels(LIBRARY)
    cfg=settings(args)
    suites=list(SUITES) if args.suite=="all" else [args.suite]
    if args.command=="check":
        data=checks(kernels,cfg,suites)
        print(json.dumps(dict(status="passed",rows={k:len(v) for k,v in data.items()}),indent=2))
        return
    if args.command=="reproduce" and not args.skip_racecheck and not shutil.which("compute-sanitizer"):
        raise RuntimeError("找不到 compute-sanitizer；安装 CUDA 工具或明确使用 --skip-racecheck。")
    output=(args.out or ROOT/"runs"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")).resolve()
    if output.exists():
        raise RuntimeError(f"结果目录已存在，不覆盖：{output}")
    output.mkdir(parents=True)
    shutil.copy2(BUILD/"build.log",output/"build.log")
    meta=dict(schema_version=2,status="running",**device,settings=cfg,suites=suites,build=provenance,
              slurm_job=os.getenv("SLURM_JOB_ID"),argv=sys.argv,source_hashes={str(f.relative_to(ROOT)):digest(f) for f in [*ROOT.glob("*.py"), *(ROOT/"csrc").glob("*.cu")]},
              racecheck="not_requested",timing="preallocated, hot reused inputs, CUDA Graph launches, randomized configuration order")
    def save(): (output/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    save()
    print(f"Results: {output}",flush=True)
    try:
        if args.command=="reproduce" and "range" in suites:
            print("Validating FTZ instructions ...",flush=True)
            validate_range(output/"ftz")
        if "mma" in suites:
            meta["instruction_evidence"]=inspect_instructions(output/"instructions.json")
        if args.command=="reproduce":
            if args.skip_racecheck:
                meta["racecheck"]="explicitly_skipped"
            else:
                command=["compute-sanitizer","--tool","racecheck","--error-exitcode","99","--log-file",str(output/"racecheck.log")]
                for name in ("unified","decay","shape_mma"):
                    command += ["--kernel-name",f"kns={name}"]
                command += [sys.executable,str(ROOT/"run.py"),"check","--suite",args.suite,"--warps",','.join(map(str,args.warps))]
                if args.quick: command.append("--quick")
                print("Checking shared-memory races ...",flush=True)
                subprocess.run(command,check=True)
                meta["racecheck"]="passed"
        print("Checking numerical results ...",flush=True)
        for name,rows in checks(kernels,cfg,suites).items(): write_csv(output/f"{name}.csv",rows)
        if "inverse" in suites: write_csv(output/"inverse_resources.csv",kernels.resources(cfg['warps']))
        for suite in suites:
            print(f"Timing {suite} ...",flush=True)
            write_csv(output/f"{suite}_timing.csv",benchmark(kernels,cfg,suite,output,device['device_uuid']))
        meta["status"]="complete";save()
        from report import render
        (output/"SUMMARY.md").write_text(render(output))
        print(f"Complete: {output/'SUMMARY.md'}",flush=True)
    except BaseException as error:
        meta["status"]="failed";meta["error"]=str(error);save()
        raise


if __name__=="__main__":
    try:
        main()
    except (RuntimeError,subprocess.CalledProcessError) as error:
        print(str(error),file=sys.stderr)
        sys.exit(1)
