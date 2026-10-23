"""AgeGuard core: system model, CSI aging, GNN, quantile safety layer and WMMSE baseline.

Used by every other script. See README.md for the mapping to the paper."""
import numpy as np, jax, jax.numpy as jnp, optax, time
from scipy.special import j0
jax.config.update("jax_enable_x64", False)

# ---------------- system parameters ----------------
P = dict(L=500.0, dmin=10.0, dmax=50.0, fc=3.5e9, Pmax_dBm=20.0, noise_dBm=-95.0,
         tau_min=0.5e-3, tau_max=2.0e-3, vmax_kmh=30.0)

def gen_instances(B, K, seed, vmax_kmh=None, p=P):
    """Return beta [B,K,K] (Tx j -> Rx k at [j,k]), h_est [B,K,K] complex, rho [B,K] (per-Rx aging)."""
    rng = np.random.default_rng(seed)
    vmax = p['vmax_kmh'] if vmax_kmh is None else vmax_kmh
    tx = rng.uniform(0, p['L'], (B, K, 2))
    d = rng.uniform(p['dmin'], p['dmax'], (B, K)); ang = rng.uniform(0, 2*np.pi, (B, K))
    rx = tx + np.stack([d*np.cos(ang), d*np.sin(ang)], -1)
    dist = np.linalg.norm(tx[:, :, None, :] - rx[:, None, :, :], axis=-1)  # [B,j,k]
    dist = np.maximum(dist, 5.0)
    pl_db = 35.3 + 37.6*np.log10(dist)
    snr_lin = 10**((p['Pmax_dBm'] - p['noise_dBm'] - pl_db)/10)  # Pmax*beta/sigma^2
    v = rng.uniform(0, vmax, (B, K))/3.6
    tau = rng.uniform(p['tau_min'], p['tau_max'], (B, K))
    fd = v*p['fc']/3e8
    rho = j0(2*np.pi*fd*tau)
    gen_instances.last_x = 2*np.pi*fd*tau
    h = (rng.standard_normal((B, K, K)) + 1j*rng.standard_normal((B, K, K)))/np.sqrt(2)
    return snr_lin.astype(np.float32), h.astype(np.complex64), rho.astype(np.float32)

def sample_true_gains(key, snr, h_est, rho, S):
    """Draw S aged-channel realizations: h = rho_k*h_est + sqrt(1-rho_k^2)*e. returns G [B,S,K,K]."""
    B, K, _ = snr.shape
    k1, k2 = jax.random.split(key)
    e = (jax.random.normal(k1, (B, S, K, K)) + 1j*jax.random.normal(k2, (B, S, K, K)))/jnp.sqrt(2.)
    r = rho[:, None, None, :]  # aging indexed by receiver k (last axis)
    h = r*h_est[:, None] + jnp.sqrt(jnp.clip(1 - r**2, 0., 1.))*e
    return snr[:, None]*jnp.abs(h)**2

def sinr_from_gains(G, p):
    """G [...,K,K] (j->k), p [...,K] in [0,1] (fraction of Pmax). SINR_k."""
    K = G.shape[-1]
    sig = jnp.diagonal(G, axis1=-2, axis2=-1)*p
    intf = jnp.einsum('...jk,...j->...k', G*(1.0 - jnp.eye(K)), p)  # masked sum: no cancellation error
    return sig/(intf + 1.0)

def order_index(eps, S, delta=None):
    """m for marginal guarantee (delta=None): floor(eps(S+1)); PAC: largest m with P[Beta(m,S+1-m)>eps]<=delta."""
    from scipy.stats import beta
    if delta is None: return max(int(np.floor(eps*(S+1))), 1)
    m = 1
    while m < S and beta.sf(eps, m+1, S-m) <= delta: m += 1
    return m

def quantile_rate(Gs, p, eps, delta=None):
    """Distribution-free safety layer: rate = m-th order statistic of sampled capacities.
    Gs [B,S,K,K], p [B,K]."""
    S = Gs.shape[1]; m = order_index(eps, S, delta)
    C = jnp.log2(1 + sinr_from_gains(Gs, p[:, None, :]))  # [B,S,K]
    Cs = jnp.sort(C, axis=1)
    return Cs[:, m-1, :]

# ---------------- AgeGuard GNN ----------------
def feats(snr, h_est, rho, age_aware=True):
    g = snr*jnp.abs(h_est)**2
    gdb = 10*jnp.log10(g + 1e-12)/10.0
    K = snr.shape[1]
    r = rho if age_aware else jnp.ones_like(rho)   # age-unaware: treat CSI as fresh
    rf = rho if age_aware else jnp.zeros_like(rho)  # aging features masked out
    exp_g = snr*(r[:, None, :]**2*jnp.abs(h_est)**2 + (1 - r[:, None, :]**2))   # mean gain given est.
    edb = 10*jnp.log10(exp_g + 1e-12)/10.0
    diag = jnp.diagonal(gdb, axis1=1, axis2=2); ediag = jnp.diagonal(edb, axis1=1, axis2=2)
    node = jnp.stack([diag - 3.0, ediag - 3.0, rf, rf**2], -1)  # [B,K,4]
    # edge j->k: interference at Rx k from Tx j, and reverse (k->j) interference, aging of both ends
    e_in = gdb - 3.0; e_out = jnp.swapaxes(gdb, 1, 2) - 3.0
    rk = jnp.broadcast_to(rf[:, None, :], gdb.shape); rj = jnp.broadcast_to(rf[:, :, None], gdb.shape)
    edge = jnp.stack([e_in, e_out, edb - 3.0, rk, rj], -1)  # [B,j,k,5]
    return node, edge

def init_mlp(key, sizes):
    ps = []
    for i in range(len(sizes)-1):
        key, k = jax.random.split(key)
        ps.append((jax.random.normal(k, (sizes[i], sizes[i+1]))*np.sqrt(2/sizes[i]), jnp.zeros(sizes[i+1])))
    return ps

def mlp(ps, x, final_act=False):
    for i, (W, b) in enumerate(ps):
        x = x @ W + b
        if i < len(ps)-1 or final_act: x = jax.nn.gelu(x)
    return x

def init_gnn(key, H=64, T=3):
    ks = jax.random.split(key, 2*T + 2)
    return dict(inp=init_mlp(ks[0], [4, H, H]),
                msg=[init_mlp(ks[1+2*t], [2*H + 5, H, H]) for t in range(T)],
                upd=[init_mlp(ks[2+2*t], [3*H, H, H]) for t in range(T)],
                out=[init_mlp(ks[-1], [H, H, 1])[0], (jnp.zeros((H, 1)), jnp.zeros(1))])  # zero-init head: p=0.5 at start

def gnn_forward(params, node, edge):
    B, K, _ = node.shape
    x = mlp(params['inp'], node, True)
    mask = 1.0 - jnp.eye(K)
    for Wm, Wu in zip(params['msg'], params['upd']):
        xj = jnp.broadcast_to(x[:, :, None, :], (B, K, K, x.shape[-1]))
        xk = jnp.broadcast_to(x[:, None, :, :], (B, K, K, x.shape[-1]))
        m = mlp(Wm, jnp.concatenate([xj, xk, edge], -1), True)  # messages j->k
        m = m*mask[None, :, :, None]
        agg_mean = m.sum(1)/(K - 1)
        agg_max = jnp.max(m - 1e4*(1-mask)[None, :, :, None], axis=1)
        x = x + mlp(Wu, jnp.concatenate([x, agg_mean, agg_max], -1))
    return jax.nn.sigmoid(mlp(params['out'], x)[..., 0])  # p in [0,1]

# ---------------- WMMSE baselines ----------------
def wmmse(G, iters=100):
    """Classic SISO-IC WMMSE (Shi et al. 2011) on gain matrix G [B,K,K] (j->k), returns p in [0,1]."""
    Hd = np.sqrt(np.diagonal(G, axis1=1, axis2=2)); v = np.ones_like(Hd)
    for _ in range(iters):
        rx = np.einsum('bjk,bj->bk', G, v**2) + 1.0
        u = Hd*v/rx
        w = 1.0/(1 - u*Hd*v + 1e-12)
        num = w*u*Hd
        den = np.einsum('bjk,bk->bj', G, w*u**2)
        v = np.clip(num/den, 0, 1)
    return v**2
