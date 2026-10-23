"""Quick demo (under 1 minute with the pretrained model): load AgeGuard, then compare it with WMMSE
on aged CSI, with and without the quantile safety layer and conformal recalibration.

    python demo.py            # uses pretrained/ageguard_main.pkl
    python demo.py --train    # trains a small model from scratch instead (1-2 min)
"""
import numpy as np, jax, jax.numpy as jnp
from core import gen_instances, feats, gnn_forward, wmmse, sinr_from_gains
from evalu import qrate_np, eval_true, est_gain
from train import train

EPS = 0.05          # outage target
K = 20              # number of links

# 1) load the pretrained paper model (pretrained/ageguard_main.pkl), or train a small one with --train
import sys, os, pickle
if '--train' in sys.argv or not os.path.exists('pretrained/ageguard_main.pkl'):
    params, _ = train(K=K, steps=300, vmax=60.0, log=True)     # quick 300-iteration model
else:
    params = jax.tree_util.tree_map(jnp.asarray, pickle.load(open('pretrained/ageguard_main.pkl', 'rb'))['params'])
    print('loaded pretrained AgeGuard model (3000 iterations, as in the paper)')

# 2) unseen test networks with users up to 30 km/h
snr, h, rho = gen_instances(50, K, seed=2024, vmax_kmh=30.0)
key = jax.random.PRNGKey(0)

def report(name, p, r):
    g, o = eval_true(snr, h, rho, p, r, jax.random.PRNGKey(1), St=500)
    act = r > 1e-3
    print(f"{name:28s} sum goodput {g.sum(1).mean():6.2f} bps/Hz   outage {o[act].mean():.3f}")

# WMMSE on aged CSI, rate from the estimated SINR (conventional)
pw = wmmse(est_gain(snr, h))
report('WMMSE, naive rate', pw, np.log2(1 + np.array(sinr_from_gains(est_gain(snr, h), pw))))
# WMMSE + quantile safety layer
report('WMMSE + safety layer', pw, qrate_np(snr, h, rho, pw, EPS, key))
# AgeGuard: age-aware GNN powers + safety layer
pg = np.array(gnn_forward(params, *feats(jnp.array(snr), jnp.array(h), jnp.array(rho))))
r_ag = qrate_np(snr, h, rho, pg, EPS, key)
report('AgeGuard', pg, r_ag)

# 3) model mismatch: the true Doppler is twice the assumed one (kappa = 2)
from scipy.special import j0
from core import sample_true_gains, gen_instances as gi
snr, h, rho = gi(100, K, seed=7, vmax_kmh=30.0); rho_true = j0(2.0 * gi.last_x).astype(np.float32)
pg = np.array(gnn_forward(params, *feats(jnp.array(snr), jnp.array(h), jnp.array(rho))))
A = qrate_np(snr, h, rho, pg, EPS, key)                                   # model-based rates
def report_true(name, r):
    g, o = eval_true(snr, h, rho_true, pg, r, jax.random.PRNGKey(1), St=500)
    act = r > 1e-3
    print(f"{name:28s} sum goodput {g.sum(1).mean():6.2f} bps/Hz   outage {o[act].mean():.3f}")
print("\nTrue Doppler = 2 x assumed Doppler:")
report_true('AgeGuard (model only)', A)

# 4) conformal recalibration (Proposition 3): c_hat from observed capacities of N calibration networks
N = 500
cs, ch, cr = gi(N, K, seed=99, vmax_kmh=30.0); cr_true = j0(2.0 * gi.last_x).astype(np.float32)
pc = np.array(gnn_forward(params, *feats(jnp.array(cs), jnp.array(ch), jnp.array(cr))))
Ac = qrate_np(cs, ch, cr, pc, EPS, key)
G = sample_true_gains(jax.random.PRNGKey(6), jnp.array(cs), jnp.array(ch), jnp.array(cr_true), 1)     # one real packet
C_obs = np.array(jnp.log2(1 + sinr_from_gains(G, jnp.array(pc)[:, None, :])))[:, 0, :]
rng = np.random.default_rng(0); scores = []
for n in range(N):
    act = np.where(Ac[n] > 1e-6)[0]
    if len(act): j = rng.choice(act); scores.append(C_obs[n, j] / Ac[n, j])   # one random active link per network
k = int(np.floor(EPS * (len(scores) + 1)))
c_hat = np.sort(scores)[k - 1]
print(f"conformal back-off factor c_hat = {c_hat:.3f}")
report_true('AgeGuard + recalibration', c_hat * A)
