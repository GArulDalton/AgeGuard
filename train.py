"""Unsupervised training of the AgeGuard GNN through the order-statistic safety layer (Algorithm 1)."""
import sys, pickle
from core import *
def make_loss(eps, S, age_aware=True):
    def loss(params, snr, h, rho, key):
        node, edge = feats(snr, h, rho, age_aware)
        p = gnn_forward(params, node, edge)
        Gs = sample_true_gains(key, snr, h, rho, S)
        r = quantile_rate(Gs, p, eps)
        return -jnp.mean(jnp.sum(r, -1))
    return loss
def train(K=20, steps=3000, B=32, S=64, eps=0.05, age_aware=True, vmax=None, seed=0, H=64, T=3, log=True):
    params = init_gnn(jax.random.PRNGKey(seed), H, T)
    sched = optax.cosine_decay_schedule(1e-3, steps, 0.05)
    opt = optax.adam(sched); st = opt.init(params)
    lf = make_loss(eps, S, age_aware)
    @jax.jit
    def step(params, st, snr, h, rho, key):
        l, g = jax.value_and_grad(lf)(params, snr, h, rho, key)
        u, st = opt.update(g, st, params); return optax.apply_updates(params, u), st, l
    key = jax.random.PRNGKey(seed+100); hist=[]
    t0=time.time()
    for it in range(steps):
        snr, h, rho = gen_instances(B, K, 10_000+it+seed*1_000_000, vmax)
        key, k = jax.random.split(key)
        params, st, l = step(params, st, snr, h, rho, k)
        if it % 50 == 0:
            hist.append((it, -float(l)))
            if log and it % 500 == 0: print(it, -float(l), round(time.time()-t0,1), flush=True)
    return params, hist
if __name__ == "__main__":
    steps=int(sys.argv[1])
    p,h = train(steps=steps)
