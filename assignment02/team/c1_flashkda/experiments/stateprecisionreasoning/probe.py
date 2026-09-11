"""Single-chunk zero-state control: separate inverse errors from carried-state rounding."""
import argparse
import json
import os
from pathlib import Path

import torch
from run import ROOT,activate,csvwrite,digest,make_inputs,metrics,recurrence


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    import flash_kda_C as ext
    torch.set_num_threads(1)
    meta=dict(status='running',job=os.getenv('SLURM_JOB_ID'),uuid=str(torch.cuda.get_device_properties(0).uuid),
              sources={p.name:digest(p) for p in [ROOT/'probe.py',ROOT/'run.py',ROOT/'reference.py']},
              extension_sha256=digest(ext.__file__))
    def save():(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    save();rows=[]
    try:
        for pattern in ['weak','correlated']:
            for logit in [-4.,0.,4.]:
                raw,initial,alog,bias,_,seeds=make_inputs(pattern,16)
                raw[4].fill_(logit);initial.zero_()
                activated=activate(*raw,initial[0].bfloat16(),alog,bias)
                ref,states=recurrence(*activated,[16])
                q,k,v,g,beta=[x.cuda().unsqueeze(0) for x in raw]
                init=initial.cuda().bfloat16();final=torch.empty_like(init);output=torch.empty_like(v)
                ws=torch.empty(ext.get_workspace_size(16,3,1),device='cuda',dtype=torch.uint8)
                ext.fwd(q,k,v,g,beta,128**-.5,output,ws,alog.cuda(),bias.cuda(),-5.,init,final,None)
                # Workspace uses separated arrays: KD,QD,KR,GT,INV,Mqk; exactly one tile/head.
                offset=3*(3*4096+512)
                inverse=ws[offset:offset+3*512].view(torch.bfloat16).reshape(3,16,16).cpu().double()
                output=output[0].cpu().double()
                aq,ak,av,ag,ab,ai=activated;cg=ag.cumsum(0)
                for h,seed in enumerate(seeds):
                    gram=torch.empty(16,16,dtype=torch.float64)
                    for i in range(16):
                        gram[i]=(ak[i,h]*ak[:,h]*(cg[i,h]-cg[:,h]).exp()).sum(-1)
                    lower=torch.tril(gram*ab[:,h,None],diagonal=-1)
                    eye=torch.eye(16,dtype=torch.float64)
                    exact=torch.linalg.solve_triangular(eye+lower,eye,upper=False,unitriangular=True)
                    poly=eye-lower
                    power=lower@lower
                    for _ in range(3):
                        poly=poly+poly@power
                        power=power@power
                    assert torch.allclose(poly,exact,atol=1e-9,rtol=1e-9)
                    assert torch.equal(ref[0],ref[1]),'No checkpoint rounding precedes first-chunk outputs'
                    error=metrics(output[:,h],ref[0,:,h])
                    rows.append(dict(pattern=pattern,beta_logit=logit,seed=seed,
                        output_relative=error['relative_l2'],output_max_abs=error['max_abs'],
                        inverse_relative=((inverse[h]-exact).norm()/exact.norm()).item(),
                        inverse_residual=(((eye+lower)@inverse[h]-eye).norm()/eye.norm()).item(),
                        lower_power8_norm=torch.linalg.matrix_norm(torch.linalg.matrix_power(lower,8)).item(),
                        state_rounding_output_relative=0.,finite=error['finite']))
        csvwrite(out/'inverse_probe.csv',rows);meta['status']='complete';save();print(out)
    except BaseException as error:
        meta.update(status='failed',error=str(error));save();raise


if __name__=='__main__':main()
