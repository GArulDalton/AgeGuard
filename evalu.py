"""Evaluation utilities: fresh true-channel draws, goodput/outage, safety-layer rates, per-instance optimizer (Q-Opt)."""
from core import *
def eval_true(snr, h, rho, p, r, key, St=1000, chunk=25):
    """Return per-link goodput mean and outage prob, averaged over fresh aged-channel draws."""
    gp=[]; out=[]
    for i in range(0, snr.shape[0], chunk):
        sl=slice(i,i+chunk); key,k=jax.random.split(key)
        G = sample_true_gains(k, jnp.array(snr[sl]), jnp.array(h[sl]), jnp.array(rho[sl]), St)
        C = jnp.log2(1+sinr_from_gains(G, jnp.array(p[sl])[:,None,:]))
        ok = (C >= jnp.array(r[sl])[:,None,:] - 1e-4)
        gp.append(np.array(jnp.mean(ok*jnp.array(r[sl])[:,None,:],1)))
        out.append(np.array(1-jnp.mean(ok,1)))
    return np.concatenate(gp), np.concatenate(out)   # [B,K]
def qrate_np(snr,h,rho,p,eps,key,S=64,chunk=50):
    rs=[]
    for i in range(0,snr.shape[0],chunk):
        sl=slice(i,i+chunk); key,k=jax.random.split(key)
        Gs=sample_true_gains(k,jnp.array(snr[sl]),jnp.array(h[sl]),jnp.array(rho[sl]),S)
        rs.append(np.array(quantile_rate(Gs,jnp.array(p[sl]),eps)))
    return np.concatenate(rs)
def est_gain(snr,h): return snr*np.abs(h)**2
def mean_gain(snr,h,rho): r=rho[:,None,:]; return snr*(r**2*np.abs(h)**2+(1-r**2))
def qopt(snr,h,rho,p0,eps,key,S=64,iters=300,lr=0.05):
    Gs=sample_true_gains(key,jnp.array(snr),jnp.array(h),jnp.array(rho),S)
    z=jnp.log(jnp.clip(jnp.array(p0),1e-4,1-1e-4))-jnp.log1p(-jnp.clip(jnp.array(p0),1e-4,1-1e-4))
    opt=optax.adam(lr); st=opt.init(z)
    f=lambda z: -jnp.sum(quantile_rate(Gs,jax.nn.sigmoid(z),eps))
    @jax.jit
    def stp(z,st):
        g=jax.grad(f)(z); u,st=opt.update(g,st,z); return optax.apply_updates(z,u),st
    for _ in range(iters): z,st=stp(z,st)
    return np.array(jax.nn.sigmoid(z))
