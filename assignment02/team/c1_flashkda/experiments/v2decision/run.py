"""Full-forward value-split challenge: correctness and paired end-to-end timing."""
import argparse,ctypes as ct,csv,hashlib,importlib.util,json,os,random,statistics,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
import torch
from build import ROOT,BUILD,build

def write(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def run(args):
    import flash_kda_C as ext
    build();torch.set_num_threads(1)
    out=(args.out or ROOT/'runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')).resolve();out.mkdir(parents=True,exist_ok=False)
    libs={p:ct.CDLL(str(BUILD/f'p{p}/kernel.so')) for p in [0,1,2,4]}
    for lib in libs.values():lib.run.argtypes=[ct.c_void_p]*12+[ct.c_int]*3+[ct.c_void_p];lib.run.restype=ct.c_int
    meta=dict(status='running',job=os.getenv('SLURM_JOB_ID'),device=str(torch.cuda.get_device_properties(0)),uuid=str(torch.cuda.get_device_properties(0).uuid),
        source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'run.py',ROOT/'build.py']},
        binaries={str(p):hashlib.sha256((BUILD/f'p{p}/kernel.so').read_bytes()).hexdigest() for p in libs},
        extension_sha256=hashlib.sha256(Path(ext.__file__).read_bytes()).hexdigest())
    def save():(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    save();checks=[];timing=[];refs=[]
    cases=[('tail',3,[15,16,17,33])]+[(f'h{h}_n{n}',h,[8192//n]*n) for h in [12,96] for n in [1,8]]+[
        ('h12_short',12,[128]),('h96_skew',96,[4608]+[512]*7),
        ('weak_check',3,[8192]),('near_check',3,[8192]),('retention_check',3,[8192])]
    if args.quick:cases=cases[:1]
    uuid=meta['uuid'];uuid=uuid if uuid.startswith('GPU-') else 'GPU-'+uuid
    clockfile=(out/'clocks.csv').open('w');monitor=subprocess.Popen(['nvidia-smi','-i',uuid,'--query-gpu=timestamp,uuid,clocks.current.sm,power.draw','--format=csv','-lms','200'],stdout=clockfile)
    try:
        for name,h,lengths in cases:
            print('Running',name,flush=True);T=sum(lengths);N=len(lengths)
            torch.manual_seed(42)
            q,k,v,g=[torch.randn(1,T,h,128,device='cuda',dtype=torch.bfloat16) for _ in range(4)]
            beta=torch.randn(1,T,h,device='cuda',dtype=torch.bfloat16)
            init=(.1*torch.randn(N,h,128,128,device='cuda')).bfloat16();alog=torch.zeros(h,device='cuda');bias=torch.zeros(h,128,device='cuda')
            if name=='weak_check':g.fill_(-8)
            if name=='near_check':g.fill_(-16);beta.fill_(-4)
            if name=='retention_check':
                q.zero_();k.zero_();q[...,0]=1;k[...,0]=1;v.zero_();g.fill_(-16);beta.fill_(-12)
                init.zero_();init[...,0]=1
            cu=torch.tensor([0]+list(__import__('itertools').accumulate(lengths)),device='cuda',dtype=torch.int64)
            outputs={};functions={};keep=[]
            for method in ['official','p0','p1','p2','p4']:
                dest=torch.full_like(v,float('nan'));state=torch.full_like(init,float('nan'))
                ws=torch.empty(ext.get_workspace_size(T,h,N),device='cuda',dtype=torch.uint8)
                outputs[method]=(dest,state);keep.append(ws)
                if method=='official':
                    def call(dest=dest,state=state,ws=ws):ext.fwd(q,k,v,g,beta,128**-.5,dest,ws,alog,bias,-5.,init,state,cu)
                else:
                    fn=libs[int(method[1:])].run
                    def call(dest=dest,state=state,ws=ws,fn=fn):
                        bt=beta.reshape(T,h).t().contiguous()
                        err=fn(*[x.data_ptr() for x in [q,k,v,g,bt,init,state,dest,ws,alog,bias,cu]],T,h,N,torch.cuda.current_stream().cuda_stream)
                        if err:raise RuntimeError(f'CUDA launch {err}')
                functions[method]=call;call();torch.cuda.synchronize()
                eo=torch.equal(dest,outputs['official'][0]);es=torch.equal(state,outputs['official'][1])
                checks.append(dict(case=name,method=method,output_exact=eo,state_exact=es,finite=bool(torch.isfinite(dest).all() and torch.isfinite(state).all())))
                assert eo and es and checks[-1]['finite'],checks[-1]
            if name=='tail':
                ref=load(ROOT.parent/'stateprecisionreasoning/reference.py','state_reference')
                ref.validate();start=0
                for n,L in enumerate(lengths):
                    activated=ref.activate(*[x[0,start:start+L].cpu() for x in [q,k,v,g,beta]],init[n].cpu(),alog.cpu(),bias.cpu())
                    gold,sref=ref.recurrence(*activated,[L])
                    for method,(dest,state) in outputs.items():
                        a=dest[0,start:start+L].cpu().double();s=state[n].cpu().double().transpose(-1,-2)
                        oe=((a-gold[0]).norm()/gold[0].norm()).item();se=((s-sref[0,0]).norm()/sref[0,0].norm()).item()
                        refs.append(dict(sequence=n,length=L,method=method,output_relative=oe,state_relative=se));assert max(oe,se)<.02
                    start+=L
            graphs={}
            for method,call in functions.items():
                for _ in range(3):call()
                torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(5):call()
                graphs[method]=graph
            for rnd in range(5):
                order=list(graphs);random.Random(rnd).shuffle(order)
                for method in order:
                    graph=graphs[method];graph.replay();samples=[]
                    for _ in range(10):
                        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                        a.record();graph.replay();b.record();b.synchronize();samples.append(a.elapsed_time(b)*200)
                    timing.append(dict(case=name,method=method,round=rnd,median_us=statistics.median(samples),min_us=min(samples),max_us=max(samples)))
            write(out/'checks.csv',checks);write(out/'reference.csv',refs);write(out/'timing.csv',timing)
        lines=['# Full forward：value-split 挑战','', '| case | official µs | p0 µs | p1 µs | p2 µs | p4 µs | best split speedup |','|---|---:|---:|---:|---:|---:|---:|']
        for name,_,_ in cases:
            values={m:statistics.median([r['median_us'] for r in timing if r['case']==name and r['method']==m]) for m in functions}
            lines.append('| '+name+' | '+' | '.join(f'{v:.3f}' for v in values.values())+f" | {values['official']/min(values['p2'],values['p4']):.3f}× |")
        (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n');meta['status']='complete';save()
    except BaseException as e:meta.update(status='failed',error=str(e));save();raise
    finally:monitor.terminate();monitor.wait();clockfile.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path);p.add_argument('--quick',action='store_true');run(p.parse_args())
