"""Algebra checks against the provided naive KDA; no performance claims."""
import importlib.util
from pathlib import Path
import math
import torch

ROOT=Path(__file__).resolve().parent
REF=ROOT.parent.parent/'fla_kda_ref/naive.py'


def relative(x,y):
    return (torch.linalg.vector_norm((x-y).double())/torch.linalg.vector_norm(y.double()).clamp_min(1e-30)).item()


def validate():
    torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('kda_reference',REF)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    naive=mod.naive_recurrent_kda
    rows=[]
    for seed in range(3):
        torch.manual_seed(seed)
        B,T,H,K,V=2,32,4,128,128
        q=torch.nn.functional.normalize(torch.randn(B,T,H,K),dim=-1)
        k=torch.nn.functional.normalize(torch.randn(B,T,H,K),dim=-1)
        v=torch.randn(B,T,H,V);g=-.1*torch.rand(B,T,H,K);beta=torch.sigmoid(torch.randn(B,T,H))
        state=.1*torch.randn(B,H,K,V)
        gold,final=naive(q,k,v,g,beta,initial_state=state,output_final_state=True)
        def record(name,o,s,expected):
            oe,se=relative(o,gold),relative(s,final)
            if expected:assert max(oe,se)<2e-5,(name,oe,se)
            else:assert max(oe,se)>.01,(name,oe,se)
            rows.append(dict(seed=seed,test=name,expected_equal=expected,output_relative_error=oe,state_relative_error=se))
        for parts in [2,4,8]:
            outputs=[];states=[]
            for j in range(parts):
                sl=slice(j*V//parts,(j+1)*V//parts)
                o,s=naive(q,k,v[...,sl],g,beta,initial_state=state[...,sl],output_final_state=True)
                outputs.append(o);states.append(s)
            record(f'value_split_{parts}',torch.cat(outputs,-1),torch.cat(states,-1),True)
        for heads in [1,2]:
            outputs=[];states=[]
            for j in range(0,H,heads):
                sl=slice(j,j+heads)
                o,s=naive(q[:,:,sl],k[:,:,sl],v[:,:,sl],g[:,:,sl],beta[:,:,sl],initial_state=state[:,sl],output_final_state=True)
                outputs.append(o);states.append(s)
            record(f'head_group_{heads}',torch.cat(outputs,2),torch.cat(states,1),True)
        for carry in [False,True]:
            outputs=[];s=state
            for j in [0,16]:
                o,s=naive(q[:,j:j+16],k[:,j:j+16],v[:,j:j+16],g[:,j:j+16],beta[:,j:j+16],
                          initial_state=s if carry or j==0 else None,output_final_state=True)
                outputs.append(o)
            record('time_split_carry' if carry else 'time_split_reset',torch.cat(outputs,1),s,carry)
        # A key-dimension split needs the complete k^T S reduction before updating any partition.
        outputs=[];states=[]
        for j in [0,64]:
            sl=slice(j,j+64)
            o,s=naive(q[...,sl],k[...,sl],v,g[...,sl],beta,scale=K**-.5,initial_state=state[:,:,sl],output_final_state=True)
            outputs.append(o);states.append(s)
        record('key_split_without_reduction',sum(outputs),torch.cat(states,-2),False)

        # Merely stacking heads' A matrices while sharing one B is not batched GEMM.
        a=torch.randn(4,16,128);b=torch.randn(4,128,16)
        true=a@b;wrong=(a.reshape(64,128)@b[0]).reshape(4,16,16)
        error=relative(wrong,true);assert error>.1
        rows.append(dict(seed=seed,test='head_stack_shared_B',expected_equal=False,output_relative_error=error,state_relative_error=''))

        # Parallel affine prefix scan is mathematically possible, but changes the algorithm.
        # Small dimensions keep this an algebra check, not a performance benchmark.
        K=8;V=8;T=32
        q=q[:,:T,:,:K].contiguous();k=k[:,:T,:,:K].contiguous()
        v=v[:,:T,:,:V].contiguous();g=g[:,:T,:,:K].contiguous();state=state[:,:,:K,:V].contiguous()
        oref,sref=naive(q,k,v,g,beta,initial_state=state,output_final_state=True)
        kk=k.double();bb=beta.double();gg=g.double()
        a=(torch.eye(K)-bb[...,None,None]*kk[..., :,None]*kk[...,None,:])*gg.exp()[...,None,:]
        b=bb[...,None,None]*kk[..., :,None]*v.double()[...,None,:]
        step=1
        while step<T:
            old_a,old_b=a,b
            a=old_a.clone();b=old_b.clone()
            a[:,step:]=old_a[:,step:]@old_a[:,:-step]
            b[:,step:]=old_a[:,step:]@old_b[:,:-step]+old_b[:,step:]
            step*=2
        all_states=a@state.double()[:,None]+b
        output=((q.double()/math.sqrt(K))[...,None,:]@all_states).squeeze(-2)
        oe,se=relative(output,oref),relative(all_states[:,-1],sref)
        assert max(oe,se)<2e-5
        rows.append(dict(seed=seed,test='affine_prefix_scan',expected_equal=True,output_relative_error=oe,state_relative_error=se))
    return rows
