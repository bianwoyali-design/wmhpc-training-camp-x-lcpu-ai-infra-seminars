"""Summarize CHUNK benchmark CSVs with the Python standard library."""
import collections
import csv
import json
from pathlib import Path
import statistics


def read(path):
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f, skipinitialspace=True))


def aggregate(rows, dimensions):
    groups = collections.defaultdict(list)
    for row in rows:
        mode = row.get('timing_mode', 'graph')
        key = tuple(row[k] for k in dimensions) + (mode,)
        us = float(row['median_us'])
        groups[key].append(us)
    return {k: (statistics.median(v), min(v), max(v), len(v)) for k, v in groups.items()}


def timing_table(rows, stage=False):
    dims = ['workload', 'chunk'] + (['stage'] if stage else []) + ['method']
    groups = aggregate(rows, dims)
    lines = ['| ' + ' | '.join(dims + ['mode', 'µs/call', 'round range µs', 'rounds']) + ' |',
             '|' + '|'.join(['---']*(len(dims)+4)) + '|']
    for key, (med, mn, mx, n) in sorted(groups.items()):
        lines.append('| ' + ' | '.join(key) + f' | {med:.3f} | {mn:.3f}–{mx:.3f} | {n} |')
    return lines


def best_table(rows, stage=False):
    # Best only within the same measured workload, stage, size and graph mode.
    dims = ['workload', 'chunk'] + (['stage'] if stage else []) + ['method']
    grouped = aggregate(rows, dims)
    candidates = collections.defaultdict(list)
    for key, values in grouped.items():
        if key[-1] != 'graph' or not key[-2].startswith(('unified', 'wmma')):
            continue
        candidates[key[:-2]].append((values[0], key[-2]))
    chosen = {k: min(v) for k, v in candidates.items()}
    head = ['workload', 'CHUNK'] + (['stage'] if stage else [])
    lines = ['| ' + ' | '.join(head + ['best candidate', 'µs/call', 'vs C=16']) + ' |',
             '|'+'|'.join(['---']*(len(head)+3))+'|']
    for key,(us,method) in sorted(chosen.items()):
        base_key = (key[0], '16', *key[2:])
        ratio = f'{us/chosen[base_key][0]:.2f}×' if base_key in chosen else '—'
        lines.append('| '+' | '.join(key)+f' | {method} | {us:.3f} | {ratio} |')
    return lines


def render(directory):
    p=Path(directory)
    meta=json.loads((p/'metadata.json').read_text()) if (p/'metadata.json').exists() else {}
    lines=['# CHUNK 实验结果', '',
           f"设备：{meta.get('device','未记录')}；状态：{meta.get('status','未记录')}；Slurm：{meta.get('slurm_job','未记录')}。",'',
           '耗时是各轮中位数的中位数；round range 是轮次中位数范围，不是置信区间。',
           '每个候选只与同一次运行、相同 workload/stage 的结果比较；不混合 eager 与 graph。','']
    present=False
    for suite in ['range','inverse','mma']:
        rows=read(p/f'{suite}_timing.csv')
        if not rows: continue
        present=True
        lines += [f'## {suite}', '']
        if suite != 'range':lines += best_table(rows,stage=suite=='mma')+['']
        lines += timing_table(rows,stage=suite=='mma')+['']
        accuracy=read(p/f'{suite}_accuracy.csv')
        if suite=='range':
            lines += ['g=-5 的范围检查（每行包含完整 batch 的因果位置）：','',
                      '| CHUNK | method | zero | Inf | NaN | max relative error |','|---:|---|---:|---:|---:|---:|']
            for r in accuracy:
                if r['case']=='constant_-5':
                    err=r['max_relative_error'] or 'nonfinite'
                    lines.append(f"| {r['chunk']} | {r['method']} | {r['zero_count']} | {r['inf_count']} | {r['nan_count']} | {err} |")
        else:
            field='max_residual' if suite=='inverse' else 'max_relative_error'
            grouped=collections.defaultdict(list)
            for r in accuracy:
                key=(r.get('case',r.get('stage')),r['chunk'],r.get('method',r.get('warps')))
                grouped[key].append(float(r[field]))
            lines += ['',f'正确性指标：{field}，下表跨种子取最大值。','',
                      '| input/stage | CHUNK | method/warps | error |','|---|---:|---|---:|']
            for key,values in sorted(grouped.items()):lines.append('| '+' | '.join(key)+f' | {max(values):.6g} |')
        lines += ['']
    if not present:
        raise ValueError(f'未找到计时 CSV：{p}')
    resources=read(p/'inverse_resources.csv')
    if resources:
        lines += ['## 求逆资源','', '| C | warps | registers/thread | shared bytes | local bytes/thread | resident CTA/SM max |',
                  '|---:|---:|---:|---:|---:|---:|']
        for r in resources:
            lines.append('| '+' | '.join([r['chunk'],r.get('warps','1'),r['registers_per_thread'],r['shared_bytes'],r['local_bytes_per_thread'],r['max_blocks_per_sm']])+' |')
    lines += ['','## 时钟与检查','']
    for f in sorted(p.glob('*clocks.csv')):
        samples=read(f)
        frequencies=[float(r['clocks.current.sm [MHz]'].split()[0]) for r in samples if r.get('clocks.current.sm [MHz]','').split()[0:1] and r['clocks.current.sm [MHz]'].split()[0].isdigit()]
        if frequencies:lines.append(f'- {f.name}: {len(frequencies)} samples, {min(frequencies):g}–{max(frequencies):g} MHz。')
    lines += [f"- racecheck: {meta.get('racecheck','见原始日志；没有日志不能视为通过')}。",'',
              '## 解释边界','',
              '- 所有微基准是预分配缓冲区、重复输入的局部算子测量，不是完整 KDA 或 K1/K2 耗时。',
              '- range 的 tile16 只修复因果衰减矩阵的构造，尚未整合进 KDA；direct 是稳定参考，不是推荐实现。',
              '- 官方求逆与统一 WMMA 的累加精度和融合组织不同，官方只作 16×16 参照。',
              '- 最佳配置只在本次 1/4/8 warp 候选内选择，不代表最优实现或算法下界。',
              '- MMA 是独立矩阵，不模拟 K2 的状态依赖；不能将四类耗时直接相加当作端到端时间。','']
    return '\n'.join(lines)
