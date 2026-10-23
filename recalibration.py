"""Conformal recalibration under model mismatch (Section IV-E / VI-I, Tables 12-13, Figs. 10-11).
Input: models.pkl. Output: recal.pkl (about 6 min)."""
"""Model-agnostic recalibration of the AgeGuard safety layer.
RC  : split rank-conformal recalibration from a small set of observed (realized) capacities.
ACI : online adaptive recalibration of the order index from ACK/NACK feedback only.
"""
import pickle, sys, time
import numpy as np, jax, jax.numpy as jnp
from scipy.special import j0
from core import gen_instances, sample_true_gains, sinr_from_gains, feats, gnn_forward, wmmse
EPS = 0.05; S = 499; K = 20; VMAX = 30.0
import os
if os.path.exists('models.pkl'):
    params = pickle.load(open('models.pkl', 'rb'))['main'][0]          # trained by train_models.py
else:
    params = jax.tree_util.tree_map(jnp.asarray, pickle.load(open('pretrained/ageguard_main.pkl', 'rb'))['params'])
gnn = jax.jit(lambda s, h, r: gnn_forward(params, *feats(s, h, r)))

def caps(key, snr, h, rho, p, n):
    """capacities of n channel draws under aging coefficients rho: [B,n,K]"""
    G = sample_true_gains(key, jnp.array(snr), jnp.array(h), jnp.array(rho), n)
    return np.array(jnp.log2(1 + sinr_from_gains(G, jnp.array(p)[:, None, :])))

def powers(snr, h, rho, method):
    if method == 'AgeGuard':
        return np.concatenate([np.array(gnn(jnp.array(snr[i:i+50]), jnp.array(h[i:i+50]), jnp.array(rho[i:i+50]))) for i in range(0, len(snr), 50)])
    return wmmse(snr * np.abs(h) ** 2)

def scenario(B, seed, kappa=1.0, hetero=False):
    snr, h, rho = gen_instances(B, K, seed, VMAX); x = gen_instances.last_x
    rng = np.random.default_rng(seed + 7)
    kap = rng.uniform(0.5, 3.0, x.shape) if hetero else kappa
    return snr, h, rho, j0(kap * x).astype(np.float32)

def model_sorted(snr, h, rho, p, key, chunk=10):
    out = []
    for i in range(0, len(snr), chunk):
        key, k = jax.random.split(key)
        out.append(np.sort(caps(k, snr[i:i+chunk], h[i:i+chunk], rho[i:i+chunk], p[i:i+chunk], S), axis=1))
    return np.concatenate(out)            # [B,S,K]

def prepare(snr, h, rho, rho_t, p, key, D=500, chunk=5):
    """anchors A = model eps-quantile rate per link [B,K]; sorted TRUE capacity draws [B,D,K]"""
    k1, k2 = jax.random.split(key)
    Cs = model_sorted(snr, h, rho, p, k1)
    A = Cs[:, M_MODEL - 1, :]
    Ct = []
    for i in range(0, len(snr), chunk):
        k2, k = jax.random.split(k2)
        Ct.append(caps(k, snr[i:i+chunk], h[i:i+chunk], rho_t[i:i+chunk], p[i:i+chunk], D))
    return A, np.concatenate(Ct)

def evaluate(A, Ct, c):
    """rates r = c*A; returns avg outage over active links and avg sum goodput (expected over true draws)"""
    r = c * A; act = A > 1e-6
    o = (Ct < r[:, None, :]).mean(1)
    per_net = np.array([o[b][act[b]].mean() for b in range(len(o)) if act[b].any()])   # network-then-link weighting (matches calibration sampling)
    return per_net.mean(), (r * (1 - o)).sum(1).mean()

def cal_scores(snr, h, rho, rho_t, p, key, chunk=10):
    """one random ACTIVE link per calibration network: s = realized capacity / model eps-quantile rate"""
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 1 << 30)))
    k1, k2 = jax.random.split(key)
    Cs = model_sorted(snr, h, rho, p, k1)
    sc = []
    for i in range(0, len(snr), chunk):
        k2, k = jax.random.split(k2)
        Ct = caps(k, snr[i:i+chunk], h[i:i+chunk], rho_t[i:i+chunk], p[i:i+chunk], 1)[:, 0, :]
        for b in range(Ct.shape[0]):
            act = np.where(Cs[i+b, M_MODEL - 1, :] > 1e-6)[0]
            if len(act) == 0: continue
            j = rng.choice(act); sc.append(Ct[b, j] / Cs[i+b, M_MODEL - 1, j])
    return np.array(sc)

def conformal_c(sc, eps=EPS):
    n = len(sc); k = int(np.floor(eps * (n + 1)))
    return 0.0 if k < 1 else float(np.sort(sc)[k - 1])

M_MODEL = int(np.floor(EPS * (S + 1)))

if __name__ == '__main__':
    R_ = {}; t0 = time.time()
    NCAL_POOL = 1000; NTEST = 300; NREP = 300
    R_['sweep'] = {}
    for sc in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 'het']:
        het = sc == 'het'; kap = 1.0 if het else sc
        st, ht, rt, rtt = scenario(NTEST, 9000, kap, het)
        s_, hc, rc, rct = scenario(NCAL_POOL, 12000, kap, het)
        res = {}
        for meth in ['WMMSE', 'AgeGuard']:
            pt = powers(st, ht, rt, meth); pc = powers(s_, hc, rc, meth)
            A, Ct = prepare(st, ht, rt, rtt, pt, jax.random.PRNGKey(1))
            pool = cal_scores(s_, hc, rc, rct, pc, jax.random.PRNGKey(2))
            o0, g0 = evaluate(A, Ct, 1.0)
            rng = np.random.default_rng(5); cs, os_, gs = [], [], []
            for _ in range(NREP):
                c = conformal_c(rng.choice(pool, 200, replace=False)); o, g = evaluate(A, Ct, c)
                cs.append(c); os_.append(o); gs.append(g)
            hold = []   # direct check of the theorem: P(s_test < c_hat) over random cal/test splits of held-out scores
            for _ in range(NREP):
                perm = rng.permutation(len(pool)); c = conformal_c(pool[perm[:200]]); hold.append(np.mean(pool[perm[200:]] < c))
            res[meth] = dict(holdout=np.mean(hold), model=(o0, g0), rc=(np.mean(os_), np.mean(gs), np.percentile(os_, 90), np.mean(cs)), pool=pool, A=A, Ct=Ct)
        R_['sweep'][sc] = {m: {k: v for k, v in d.items() if k not in ('A', 'Ct')} for m, d in res.items()}
        if sc == 2.0: keep = res
        print('sweep', sc, {m: (round(float(d['model'][0]), 3), round(float(d['model'][1]), 1), round(float(d['rc'][0]), 3), round(float(d['rc'][1]), 1), round(float(d['rc'][3]), 3), round(float(d['holdout']), 4)) for m, d in res.items()}, round(time.time() - t0), flush=True)
        pickle.dump(R_, open('recal.pkl', 'wb'))
    R_['ncal'] = {}
    A, Ct, pool = keep['AgeGuard']['A'], keep['AgeGuard']['Ct'], keep['AgeGuard']['pool']
    rng = np.random.default_rng(11)
    for n in [20, 50, 100, 200, 500]:
        os_, gs = [], []
        for _ in range(500):
            o, g = evaluate(A, Ct, conformal_c(rng.choice(pool, n, replace=False))); os_.append(o); gs.append(g)
        R_['ncal'][n] = dict(out_mean=np.mean(os_), out_p90=np.percentile(os_, 90), frac_viol=np.mean(np.array(os_) > EPS), gp=np.mean(gs), gp_std=np.std(gs))
        print('ncal', n, {k: round(float(v), 4) for k, v in R_['ncal'][n].items()}, flush=True)
    pickle.dump(R_, open('recal.pkl', 'wb'))
    # ---------- online ACK/NACK adaptation (log-multiplier tracking) under drifting mismatch ----------
    T = 900; eta = 0.05
    kap_t = np.where(np.arange(T) < 300, 1.0, np.where(np.arange(T) < 600, 2.0, 0.5))
    st, ht, rt, _ = scenario(T, 30000, 1.0); x = gen_instances.last_x
    rtt = np.stack([j0(kap_t[i] * x[i]) for i in range(T)]).astype(np.float32)
    pt = powers(st, ht, rt, 'AgeGuard')
    key = jax.random.PRNGKey(21); th = 0.0
    tr = {'model': [], 'aci': [], 'c': [], 'gp_model': [], 'gp_aci': []}
    for t in range(T):
        key, k1, k2, k3 = jax.random.split(key, 4)
        A_ = np.sort(caps(k1, st[t:t+1], ht[t:t+1], rt[t:t+1], pt[t:t+1], S), axis=1)[0, M_MODEL - 1]
        Ct_ = caps(k2, st[t:t+1], ht[t:t+1], rtt[t:t+1], pt[t:t+1], 200)[0]
        Cobs = caps(k3, st[t:t+1], ht[t:t+1], rtt[t:t+1], pt[t:t+1], 1)[0, 0]
        act = A_ > 1e-6; c = float(np.exp(th))
        for name, cc in [('model', 1.0), ('aci', c)]:
            r = cc * A_; o = (Ct_ < r[None, :]).mean(0)
            tr[name].append(o[act].mean() if act.any() else 0.0); tr['gp_' + name].append(float((r * (1 - o)).sum()))
        nack = (Cobs < c * A_)[act].mean() if act.any() else 0.0
        th = float(np.clip(th + eta * (EPS - nack), np.log(0.05), np.log(3.0))); tr['c'].append(c)
    R_['aci'] = {k: np.array(v) for k, v in tr.items()}; R_['aci']['kappa'] = kap_t
    seg = lambda v: [round(float(np.mean(v[a:b])), 4) for a, b in [(100, 300), (400, 600), (700, 900)]]
    print('aci', {k: seg(tr[k]) for k in ['model', 'aci', 'gp_model', 'gp_aci', 'c']}, round(time.time() - t0), flush=True)
    pickle.dump(R_, open('recal.pkl', 'wb'))
    print('done', flush=True)
