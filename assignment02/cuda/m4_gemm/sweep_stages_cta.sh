#!/usr/bin/env bash
# 问题 4.4(a) cta_group::2 的 stages 扫描：S ∈ {2,3,4,6}，测试形状
# 与 4.3 保持一致，便于比较 shared memory 减少后的 stage 敏感度。
# 用法：./sweep_stages_cta.sh（可在任意目录执行）。
set -u

cd "$(dirname "${BASH_SOURCE[0]}")"

# 没有现成的 Slurm allocation 时，为整轮扫描申请一次 GPU，避免每个
# shape/stage 组合分别排队；若已经通过 salloc/srun 进入节点则直接运行。
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    script_path="$PWD/$(basename "${BASH_SOURCE[0]}")"
    exec srun -p gpu --gres=gpu:1 --time=00:20:00 --ntasks=1 "$script_path" "$@"
fi

for shape in "4096 4096 4096" "256 4096 16384"; do
    echo "== 4.4 CTA group::2，形状 $shape =="
    for s in 2 3 4 6; do
        echo "-- STAGES=$s --"
        (cd .. && STAGES="$s" make -sB bin/m4_gemm/04_cta) || exit 1
        timeout -k 5 180 ../bin/m4_gemm/04_cta $shape ||
            echo "S=$s: 失败或挂死（超时被杀）"
    done
done
