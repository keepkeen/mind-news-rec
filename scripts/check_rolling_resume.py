"""Check exact next-update recovery on a tiny CUDA model without exporting a production checkpoint."""
from pathlib import Path
import tempfile
import shutil
import torch
from torch import nn
import mid_checkpoint as mid
from jobio import CACHE,EXPORT,write_json

class Encoder(nn.Linear):
    def save_trainable(self):return {n:p.detach().cpu() for n,p in self.named_parameters()}
def run():
    torch.manual_seed(123)
    e=Encoder(3,4).cuda();h=nn.Linear(4,1).cuda()
    opt=torch.optim.AdamW([*e.parameters(),*h.parameters()],lr=.002,fused=True)
    sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:1/(1+step*.03))
    base=torch.arange(36,device="cuda",dtype=torch.float32).reshape(12,3)/36
    order_rng=torch.cuda.get_rng_state();order=torch.randperm(len(base),device="cuda")
    def update(ids):
        opt.zero_grad();x=base[ids]+torch.rand(len(ids),3,device="cuda")+.01*torch.rand(1).item()
        loss=h(e(x)).square().mean();loss.backward();opt.step();sch.step()
        return loss.detach().cpu(),[p.detach().cpu().clone() for p in [*e.parameters(),*h.parameters()]]
    update(order[:4])
    spec=dict(revision="fixture",training_splits=["train"],positive_pairs=12,logical_positives=4,max_tokens=64,epochs=3,shadow=False)
    with tempfile.TemporaryDirectory(dir=CACHE) as tmp:
        out=Path(tmp)/"saved.pt";old=mid.export_file
        try:
            mid.export_file=lambda p:shutil.copyfile(p,out)
            mid.snapshot("selfcheck",1,1,order_rng,e,h,opt,sch,spec,123,0.,0)
        finally:mid.export_file=old
        loss1,p1=update(order[4:8])
        ep,it,seedstate,_,_=mid.restore(out,"selfcheck",e,h,opt,sch,spec,123)
        g=torch.Generator(device="cuda");g.set_state(seedstate)
        restored_order=torch.randperm(len(base),device="cuda",generator=g)
        loss2,p2=update(restored_order[4:8])
        assert (ep,it)==(1,1) and torch.equal(order,restored_order) and torch.equal(loss1,loss2) and all(torch.equal(a,b) for a,b in zip(p1,p2))
    write_json(EXPORT/"rolling_resume_selfcheck.json",dict(same_permutation=True,next_loss_bitwise_equal=True,next_parameters_bitwise_equal=True,
               restored_optimizer=True,restored_scheduler=True,restored_cpu_and_cuda_rng=True))
if __name__=="__main__":run()
