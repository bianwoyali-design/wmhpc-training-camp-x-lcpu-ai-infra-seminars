"""Discussion 4: reproducible per-stage Nsight Compute bottleneck evidence."""
import argparse
import collections
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'parallelismreasoning'
CASES={'h12_n1':(12,[8192],False),'h12_n8':(12,[1024]*8,False),
       'h96_n1':(96,[8192],False),'h96_n8':(96,[1024]*8,False),
       'h96_skew8':(96,[4608]+[512]*7,True)}
METRICS=[
'gpu__time_duration.sum','sm__throughput.avg.pct_of_peak_sustained_elapsed',
'dram__throughput.avg.pct_of_peak_sustained_elapsed','dram__bytes_read.sum','dram__bytes_write.sum',
'lts__throughput.avg.pct_of_peak_sustained_elapsed','lts__t_sector_hit_rate.pct','lts__t_sectors.sum',
'l1tex__throughput.avg.pct_of_peak_sustained_elapsed',
'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active',
'sm__issue_active.avg.pct_of_peak_sustained_elapsed',
'sm__cycles_active.avg.pct_of_peak_sustained_elapsed','sm__cycles_active.min','sm__cycles_active.max',
'sm__warps_active.avg.pct_of_peak_sustained_active','smsp__warps_eligible.avg.per_cycle_active',
'smsp__issue_active.avg.pct_of_peak_sustained_active','gpc__cycles_elapsed.avg.per_second',
'sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32.sum','sm__ops_path_tensor_op_hmma_src_fp16_dst_fp16.sum',
'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum','l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
]+[f'sm__inst_executed_pipe_{pipe}.avg.pct_of_peak_sustained_elapsed' for pipe in ['alu','lsu','adu','xu','fma']
]+[f'smsp__warp_issue_stalled_{kind}_per_warp_active.pct' for kind in
   ['short_scoreboard','long_scoreboard','barrier','membar','wait','math_pipe_throttle','mio_throttle','not_selected']]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_csv(path,rows):
    with Path(path).open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def inputs(case):
    import torch
    import flash_kda_C as ext
    h,lengths,varlen=CASES[case];n=len(lengths);t=sum(lengths) if varlen else lengths[0];b=1 if varlen else n
    torch.manual_seed(1234)
    dims=(b,t,h,128)
    q=torch.nn.functional.normalize(torch.randn(dims,device='cuda'),dim=-1).bfloat16()
    k=torch.nn.functional.normalize(torch.randn(dims,device='cuda'),dim=-1).bfloat16()
    v=torch.randn(dims,device='cuda',dtype=torch.bfloat16);g=torch.randn_like(v)
    beta=torch.randn(b,t,h,device='cuda',dtype=torch.bfloat16)
    alog=torch.rand(h,device='cuda');bias=torch.rand(h,128,device='cuda')
    init=(.1*torch.randn(n,h,128,128,device='cuda')).bfloat16();final=torch.empty_like(init);out=torch.empty_like(v)
    workspace=torch.empty(ext.get_workspace_size(sum(lengths),h,n),device='cuda',dtype=torch.uint8)
    cu=torch.tensor([0]+list(__import__('itertools').accumulate(lengths)),device='cuda',dtype=torch.int64) if varlen else None
    def call():ext.fwd(q,k,v,g,beta,128**-.5,out,workspace,alog,bias,-5.,init,final,cu)
    return call,out,final

def target(case,path):
    import torch
    torch.set_num_threads(1)
    call,out,state=inputs(case)
    for _ in range(10):call()
    torch.cuda.synchronize();gold=out.clone();sref=state.clone()
    torch.cuda.profiler.start();call();torch.cuda.synchronize();torch.cuda.profiler.stop()
    assert torch.isfinite(out).all() and torch.isfinite(state).all()
    assert torch.equal(out,gold) and torch.equal(state,sref)
    if path:path.write_text(json.dumps(dict(case=case,repeat_output_exact=True,repeat_state_exact=True))+'\n')


def normal_timing(case,out):
    import torch
    spec=importlib.util.spec_from_file_location('parallelism_helpers',PREVIOUS/'run.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);helper.build()
    h,lengths,varlen=CASES[case]
    graph,control,keep=helper.prepare_case(h,lengths,varlen,10)
    rows=[];rng=random.Random(42)
    for rid in range(5):
        modes=['full','k1','k2'];rng.shuffle(modes)
        for mode in modes:
            control.mode(mode);graph.replay();torch.cuda.synchronize();times=[]
            for _ in range(5):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record();graph.replay();end.record();end.synchronize()
                times.append(start.elapsed_time(end)*1000/10)
            rows.append(dict(case=case,mode=mode,round=rid,median_us=statistics.median(times),min_us=min(times),max_us=max(times)))
    return rows


def parse_raw(path):
    text=path.read_text();lines=text.splitlines()
    start=next(i for i,line in enumerate(lines) if line.startswith('"ID"'))
    data=list(csv.DictReader(io.StringIO('\n'.join(lines[start:]))))
    if data and 'Metric Name' in data[0]:
        return [r for r in data if r.get('Metric Name')]
    # NCU 2025.3 raw CSV is wide, with a separate units row.
    units=data[0]
    return [{'Kernel Name':r['Kernel Name'],'Metric Name':name,'Metric Unit':units.get(name,''),'Metric Value':value}
            for r in data[1:] for name,value in r.items() if '__' in name and value not in ('',None)]


def report(out):
    normalized=[];summaries=[]
    for raw in sorted(out.glob('*.raw.csv')):
        label=raw.name.removesuffix('.raw.csv');case,cache=label.rsplit('__',1)
        rows=parse_raw(raw);groups=collections.defaultdict(dict)
        for r in rows:
            name=r['Kernel Name'];stage='k1' if '_prepare' in name else 'k2' if '_recurrence' in name else None
            if stage is None:continue
            normalized.append(dict(case=case,cache=cache,stage=stage,metric=r['Metric Name'],unit=r['Metric Unit'],value=r['Metric Value']))
            try:groups[stage][r['Metric Name']]=float(r['Metric Value'].replace(',',''))
            except ValueError:pass
        assert set(groups)=={'k1','k2'},(raw,groups.keys())
        for stage,m in groups.items():
            for key in METRICS:
                if key not in m:raise RuntimeError(f'{label}/{stage}: missing metric {key}')
            duration=m['gpu__time_duration.sum'];seconds=duration*1e-9
            dram=m['dram__bytes_read.sum']+m['dram__bytes_write.sum']
            flop=m['sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32.sum']+m['sm__ops_path_tensor_op_hmma_src_fp16_dst_fp16.sum']
            summaries.append(dict(case=case,cache=cache,stage=stage,time_us=duration/1000,
                sm_pct=m['sm__throughput.avg.pct_of_peak_sustained_elapsed'],dram_pct=m['dram__throughput.avg.pct_of_peak_sustained_elapsed'],
                l2_pct=m['lts__throughput.avg.pct_of_peak_sustained_elapsed'],l1_pct=m['l1tex__throughput.avg.pct_of_peak_sustained_elapsed'],
                tensor_elapsed_pct=m['sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed'],
                tensor_active_pct=m['sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active'],
                sm_active_pct=m['sm__cycles_active.avg.pct_of_peak_sustained_elapsed'],
                eligible=m['smsp__warps_eligible.avg.per_cycle_active'],issue_pct=m['smsp__issue_active.avg.pct_of_peak_sustained_active'],
                occupancy_pct=m['sm__warps_active.avg.pct_of_peak_sustained_active'],
                l2_hit_pct=m['lts__t_sector_hit_rate.pct'],dram_bytes=dram,dram_GBs=dram/seconds/1e9,
                tensor_ops=flop,tensor_TFLOPs=flop/seconds/1e12,tensor_ops_per_dram_byte=flop/dram if dram else '',
                clock_MHz=m['gpc__cycles_elapsed.avg.per_second']/1e6,
                short_scoreboard_pct=m['smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct'],
                long_scoreboard_pct=m['smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct'],
                wait_pct=m['smsp__warp_issue_stalled_wait_per_warp_active.pct'],
                barrier_pct=m['smsp__warp_issue_stalled_barrier_per_warp_active.pct'],
                math_throttle_pct=m['smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct'],
                mio_throttle_pct=m['smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct']))
    write_csv(out/'metrics.csv',normalized);write_csv(out/'summary.csv',summaries)
    lines=['# NCU 分阶段指标','',
        'cold：kernel replay、清缓存；warm：application replay、不清缓存。吞吐百分比均为对应单位的峰值归一化。',
        'Tensor/SM/DRAM/L2/L1 为 elapsed 口径；occupancy、issue 和 eligible 为 active 口径。','',
        '| case | cache | stage | µs | SM % | Tensor % | DRAM % | L2 % | L1 % | eligible |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append('| '+' | '.join(str(s[k]) if isinstance(s[k],str) else f'{s[k]:.2f}' for k in
            ['case','cache','stage','time_us','sm_pct','tensor_elapsed_pct','dram_pct','l2_pct','l1_pct','eligible'])+' |')
    (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')


def reproduce(args):
    import torch,flash_kda_C
    torch.set_num_threads(1)
    if torch.cuda.get_device_capability()!=(10,3):raise RuntimeError('需要 B300')
    out=(args.out or ROOT/'runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')).resolve();out.mkdir(parents=True,exist_ok=False)
    dev=torch.cuda.get_device_properties(0);uuid=str(dev.uuid)
    if not uuid.startswith('GPU-'):uuid='GPU-'+uuid
    names=[args.case] if args.case else list(CASES)
    caches=['cold'] if args.quick else ['cold','warm']
    meta=dict(status='running',device=dev.name,uuid=uuid,sm_count=dev.multi_processor_count,slurm_job=os.getenv('SLURM_JOB_ID'),
        torch=torch.__version__,cuda=torch.version.cuda,ncu=subprocess.check_output(['ncu','--version'],text=True),
        cases=names,cache_modes=caches,metrics=METRICS,clock_control='none',source_sha256=digest(ROOT/'run.py'),
        extension=flash_kda_C.__file__,extension_sha256=digest(flash_kda_C.__file__),helper_sha256=digest(PREVIOUS/'run.py'),commands=[])
    def save():(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    save();normal=[]
    with (out/'clocks.csv').open('w') as clocklog:
        monitor=subprocess.Popen(['nvidia-smi','-i',uuid,'--query-gpu=timestamp,uuid,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu','--format=csv','-lms','200'],stdout=clocklog,stderr=subprocess.DEVNULL)
        try:
            for case in names:
                normal+=normal_timing(case,out);write_csv(out/'normal_timing.csv',normal)
                for cache in caches:
                    label=f'{case}__{cache}'
                    cmd=['ncu','--target-processes','all','--profile-from-start','off','--kernel-name-base','function','--kernel-name','regex:_flash_kda_fwd_(prepare|recurrence)',
                        '--launch-count','2','--clock-control','none','--cache-control','all' if cache=='cold' else 'none',
                        '--replay-mode','kernel' if cache=='cold' else 'application',
                        '--section','LaunchStats','--section','Occupancy','--metrics',','.join(METRICS),
                        '--export',str(out/label),sys.executable,str(ROOT/'run.py'),'target','--case',case,'--out',str(out/f'{label}.check.json')]
                    meta['commands'].append(cmd);save();print(f'Profiling {label}',flush=True)
                    with (out/f'{label}.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=log,check=True)
                    with (out/f'{label}.raw.csv').open('w') as csvfile:
                        subprocess.run(['ncu','--import',str(out/f'{label}.ncu-rep'),'--page','raw','--csv','--print-units','base'],stdout=csvfile,stderr=subprocess.PIPE,check=True)
            report(out);meta['status']='complete';save();print(out,flush=True)
        except BaseException as error:
            meta.update(status='failed',error=str(error));save();raise
        finally:monitor.terminate();monitor.wait()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['target','reproduce','report'])
    p.add_argument('--case',choices=list(CASES));p.add_argument('--quick',action='store_true');p.add_argument('--out',type=Path)
    args=p.parse_args()
    if args.command=='target':
        if not args.case:p.error('target requires --case')
        target(args.case,args.out)
    elif args.command=='report':
        if args.out is None:p.error('report requires --out')
        report(args.out)
    else:reproduce(args)

if __name__=='__main__':main()
