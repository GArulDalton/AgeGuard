"""Per-decision latency benchmark (Table 9). Input: models.pkl. Output: bench.pkl."""
import pickle
from core import *; from evalu import *
M=pickle.load(open('models.pkl','rb')); pm=M['main'][0]
out={}
for K in [10,20,40]:
    snr,h,rho=gen_instances(30,K,4242,30.)
    f=jax.jit(lambda s,hh,r: gnn_forward(pm,*feats(s,hh,r)))
    def q(s,hh,r,p,key): return quantile_rate(sample_true_gains(key,s,hh,r,64),p,0.05)
    qj=jax.jit(q)
    s1,h1,r1=jnp.array(snr[:1]),jnp.array(h[:1]),jnp.array(rho[:1])
    p=f(s1,h1,r1).block_until_ready(); qj(s1,h1,r1,p,jax.random.PRNGKey(0)).block_until_ready()
    tg=[];tq=[];tw=[]
    for i in range(30):
        s1,h1,r1=jnp.array(snr[i:i+1]),jnp.array(h[i:i+1]),jnp.array(rho[i:i+1])
        t=time.perf_counter(); p=f(s1,h1,r1).block_until_ready(); tg.append(time.perf_counter()-t)
        t=time.perf_counter(); qj(s1,h1,r1,p,jax.random.PRNGKey(i)).block_until_ready(); tq.append(time.perf_counter()-t)
        t=time.perf_counter(); wmmse(est_gain(snr[i:i+1],h[i:i+1])); tw.append(time.perf_counter()-t)
    Gs=sample_true_gains(jax.random.PRNGKey(1),s1,h1,r1,64); z=jnp.zeros((1,K)); opt=optax.adam(0.05); st=opt.init(z)
    fl=lambda z: -jnp.sum(quantile_rate(Gs,jax.nn.sigmoid(z),0.05))
    @jax.jit
    def stp(z,st):
        g=jax.grad(fl)(z); u,st=opt.update(g,st,z); return optax.apply_updates(z,u),st
    z,st=stp(z,st); z.block_until_ready(); t=time.perf_counter()
    for _ in range(300): z,st=stp(z,st)
    z.block_until_ready(); tqo=time.perf_counter()-t
    out[K]=dict(gnn=np.median(tg)*1e3,safety=np.median(tq)*1e3,wmmse=np.median(tw)*1e3,qopt=tqo*1e3)
    print(K,out[K],flush=True)
pickle.dump(out,open('bench.pkl','wb'))
