"""Generate a bounded value-split patch against the pinned official K2."""
from pathlib import Path
import hashlib,json,subprocess

ROOT=Path(__file__).resolve().parent
OFFICIAL=ROOT.parents[1]/'FlashKDA'
BUILD=ROOT/'build'

def replace(s,a,b):
    assert s.count(a)==1,(a,s.count(a))
    return s.replace(a,b)

def build():
    BUILD.mkdir(exist_ok=True)
    kernel=(OFFICIAL/'csrc/smxx/fwd_kernel2.cuh').read_text()
    launch=(OFFICIAL/'csrc/smxx/fwd_launch.cu').read_text().split('// Explicit instantiations')[0]
    wrapper=r'''
extern "C" int run(void* q,void* k,void* v,void* g,void* beta,void* init,void* final,
 void* out,void* ws,void* alog,void* bias,void* cu,int T,int H,int N,void* stream) {
 using B=cutlass::bfloat16_t;
 launch_fwd<128,true,true,false,true>((B*)q,(B*)k,(B*)v,(B*)g,(B*)beta,init,
  0.08838834764831845f,final,(B*)out,ws,(T+15)/16+N,T,H,N,(int64_t*)cu,
  (float*)alog,(float*)bias,float(-5.0*1.4426950408889634),(cudaStream_t)stream);
 return (int)cudaGetLastError();
}
'''
    for parts in [0,1,2,4]:
        directory=BUILD/f'p{parts}';directory.mkdir(exist_ok=True)
        k=kernel;l=launch
        if parts:
            k=replace(k,'constexpr int kComputeThreads = 128;',f'constexpr int kComputeThreads = {128//parts};')
            k=replace(k,'const int warp_id = compute_tid / 32;',f'const int warp_id = compute_tid / 32 + blockIdx.z * {4//parts};')
            k=replace(k,'warp_role, kComputeThreads, 1','warp_role, kComputeThreads, 32')
            k=replace(k,'cutlass::bfloat16_t* out_raw_ptr,','cutlass::bfloat16_t* out_raw_ptr,\n    cutlass::bfloat16_t* final_raw_ptr,')
            begin=k.index('    if (warp_role == WarpRole::STORE && lane_predicate) {',k.index('// --- MMA warps'))
            end=k.rindex('    __syncthreads();')
            k=k[:begin]+f'''
    if (warp_role == WarpRole::STORE) {{
        constexpr int Width = D/{parts};
        int lane=threadIdx.x%32, first=blockIdx.z*Width;
        StorePipelineState out_read;
        for(int t=0;t<t_tiles;++t) {{
            store_pipeline.consumer_wait(out_read);
            Tensor s_out=make_tensor(make_smem_ptr(shared_storage.output[out_read.index()].out.begin()),VOLayout{{}});
            int count=min(CHUNK,seq_len-t*CHUNK);
            for(int i=lane;i<count*Width;i+=32) {{
                int row=i/Width,col=first+i%Width;
                out_raw_ptr[((bos+t*CHUNK+row)*H+head_idx)*D+col]=s_out(row,col);
            }}
            __syncwarp();
            store_pipeline.consumer_release(out_read);
            ++out_read;
        }}
        Tensor state=make_tensor(make_smem_ptr(shared_storage.state_acc.begin()),StateSmemLayout{{}});
        for(int i=lane;i<Width*D;i+=32) {{
            int row=first+i/D,col=i%D;
            final_raw_ptr[((int64_t(seq_idx)*H+head_idx)*D+row)*D+col]=state(row,col);
        }}
    }}

'''+k[end:]
            l=replace(l,'constexpr int kK2Threads = 32 * 2 + 128;',f'constexpr int kK2Threads = 64 + {128//parts};')
            l=replace(l,'dim3 grid_k2(N, H);',f'dim3 grid_k2(N, H, {parts});')
            l=replace(l,'out_ptr, T_total, H, N, cu_seqlens_ptr, total_tiles','out_ptr, static_cast<BF16*>(final_state_ptr), T_total, H, N, cu_seqlens_ptr, total_tiles')
        (directory/'fwd_kernel2.cuh').write_text(k)
        source=directory/'launch.cu';source.write_text(l+wrapper)
        cmd=['nvcc','-O3','-std=c++17','-gencode','arch=compute_103a,code=sm_103a','--shared','-Xcompiler','-fPIC',
             '--expt-relaxed-constexpr','--expt-extended-lambda','--use_fast_math','-lineinfo',
             '--ptxas-options=-v,--register-usage-level=10,--warn-on-spills',
             '-I'+str(OFFICIAL/'csrc'),'-I'+str(OFFICIAL/'csrc/smxx'),'-I'+str(OFFICIAL/'cutlass/include'),
             '-I'+str(OFFICIAL/'cutlass/tools/util/include'),str(source),'-o',str(directory/'kernel.so')]
        stamp=hashlib.sha256((k+l+wrapper+str(cmd)).encode()).hexdigest()
        if not (directory/'stamp').exists() or (directory/'stamp').read_text()!=stamp:
            print('Building',parts,flush=True)
            with (directory/'build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=f,check=True)
            (directory/'stamp').write_text(stamp)
    return BUILD

if __name__=='__main__':build()
