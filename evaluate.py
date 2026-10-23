"""Run all main experiments (Tables 2-11, Figs. 3-9). Input: models.pkl. Output: results.pkl (about 20 min)."""
import pickle
from train import *; from evalu import *
R={}
def save(): pickle.dump(R,open('results.pkl','wb'))
t=time.time()
M=pickle.load(open('models.pkl','rb'))
pm,pn,pa,p1=[M[k][0] for k in ['main','noaoi','nofeat','T1']]
for k in M: R['hist_'+k]=M[k][1]
def gnnp(params,snr,h,rho,aa=True,chunk=50):
    out=[]
    for i in range(0,snr.shape[0],chunk):
        sl=slice(i,i+chunk); n,e=feats(jnp.array(snr[sl]),jnp.array(h[sl]),jnp.array(rho[sl]),aa)
        out.append(np.array(gnn_forward(params,n,e)))
    return np.concatenate(out)

def run_methods(snr,h,rho,eps=0.05,S=64,rho_true=None,delta=None,do_qopt=True,extra=True):
    key=jax.random.PRNGKey(7); B,K,_=snr.shape; res={}; tm={}
    qr=lambda p: qrate_np(snr,h,rho,p,eps,key,S=S) if delta is None else qrate_pac(p)
    def qrate_pac(p):
        Gs=sample_true_gains(key,jnp.array(snr),jnp.array(h),jnp.array(rho),S)
        return np.array(quantile_rate(Gs,jnp.array(p),eps,delta))
    pf=np.ones((B,K),np.float32); ge=est_gain(snr,h)
    res['FP-Naive']=(pf,np.log2(1+np.array(sinr_from_gains(ge,pf))))
    t0=time.time(); pw=wmmse(ge); tm['WMMSE']=(time.time()-t0)/B
    res['WMMSE-Naive']=(pw,np.log2(1+np.array(sinr_from_gains(ge,pw))))
    res['FP-Q']=(pf,qr(pf)); res['WMMSE-Q']=(pw,qr(pw))
    t0=time.time(); pe=wmmse(mean_gain(snr,h,rho)); tm['EWMMSE']=(time.time()-t0)/B
    res['EWMMSE-Q']=(pe,qr(pe))
    if extra:
        res['GNN-PerfCSI-Q']=(gnnp(pn,snr,h,rho,False),None); res['GNN-PerfCSI-Q']=(res['GNN-PerfCSI-Q'][0],qr(res['GNN-PerfCSI-Q'][0]))
    gnnp(pm,snr[:2],h[:2],rho[:2]); t0=time.time(); pg=gnnp(pm,snr,h,rho); tm['AgeGuard']=(time.time()-t0)/B
    res['AgeGuard']=(pg,qr(pg))
    if do_qopt:
        t0=time.time(); pq=qopt(snr,h,rho,pe,eps,jax.random.PRNGKey(3),S=S); tm['Q-Opt']=(time.time()-t0)/B
        res['Q-Opt']=(pq,qr(pq))
    rt = rho if rho_true is None else rho_true
    out={}
    for n,(p,r) in res.items():
        g,o=eval_true(snr,h,rt,p,r,jax.random.PRNGKey(11))
        act=r>1e-3
        out[n]=dict(gp=g.sum(1).mean(), gp_std=g.sum(1).std(), out=o[act].mean(), p95=np.percentile(o[act],95),
                    viol=(o[act]>eps).mean(), link_gp=g.ravel(), link_out=o[act].ravel(), meanP=p.mean(), active=act.mean())
    return out,tm

B=200
snr,h,rho=gen_instances(B,20,777,30.0)
R['main'],R['time']=run_methods(snr,h,rho); save(); print('main eval',time.time()-t,flush=True)
# ablations at main setting
R['abl']={}
for nm,prm,aa in [('Full AgeGuard',pm,True),('w/o age features',pa,False),('1 MP layer',p1,True)]:
    n,e=feats(jnp.array(snr),jnp.array(h),jnp.array(rho),aa)
    p=gnnp(prm,snr,h,rho,aa); r=qrate_np(snr,h,rho,p,0.05,jax.random.PRNGKey(7))
    g,o=eval_true(snr,h,rho,p,r,jax.random.PRNGKey(11)); R['abl'][nm]=(g.sum(1).mean(),o[r>1e-3].mean())
save(); print('abl',time.time()-t,flush=True)
# PAC variant
R['pac'],_=run_methods(snr,h,rho,delta=0.1,do_qopt=False,extra=False); save()
# Doppler sweep
R['vsweep']={}
for v in [0.,10.,20.,30.,45.,60.]:
    s2,h2,r2=gen_instances(B,20,5000+int(v),v); R['vsweep'][v],_=run_methods(s2,h2,r2,do_qopt=False)
    print('v',v,time.time()-t,flush=True)
save()
# K sweep
R['ksweep']={}; R['ktime']={}
for K in [10,20,30,40]:
    s2,h2,r2=gen_instances(100,K,6000+K,30.); R['ksweep'][K],R['ktime'][K]=run_methods(s2,h2,r2,do_qopt=True)
    print('K',K,time.time()-t,flush=True)
save()
# eps sweep with S=199
R['esweep']={}
s2,h2,r2=gen_instances(100,20,8000,30.)
for e in [0.01,0.02,0.05,0.10]:
    R['esweep'][e],_=run_methods(s2,h2,r2,eps=e,S=199,do_qopt=False)
save(); print('eps',time.time()-t,flush=True)
# S sweep
R['ssweep']={}
for S in [20,64,199,499]:
    R['ssweep'][S],_=run_methods(s2,h2,r2,S=S,do_qopt=False,extra=False)
save(); print('S',time.time()-t,flush=True)
# Doppler mismatch: true speed = kappa * assumed
R['mis']={}
s2,h2,r2=gen_instances(100,20,9000,30.); x=gen_instances.last_x
from scipy.special import j0
for kap in [0.5,1.0,1.5,2.0]:
    rt=j0(kap*x).astype(np.float32); R['mis'][kap],_=run_methods(s2,h2,r2,rho_true=rt,do_qopt=False)
save(); print('done',time.time()-t,flush=True)
