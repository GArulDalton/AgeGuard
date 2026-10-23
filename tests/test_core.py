"""Basic checks of the implementation (run: python tests/test_core.py  or  pytest tests/).
Takes about 30 s on a CPU."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
from core import gen_instances, sample_true_gains, sinr_from_gains, quantile_rate, order_index, feats, init_gnn, gnn_forward, wmmse


def test_shapes():
    snr, h, rho = gen_instances(4, 10, seed=0)
    assert snr.shape == (4, 10, 10) and h.shape == (4, 10, 10) and rho.shape == (4, 10)
    p = gnn_forward(init_gnn(jax.random.PRNGKey(0)), *feats(jnp.array(snr), jnp.array(h), jnp.array(rho)))
    assert p.shape == (4, 10) and float(p.min()) >= 0 and float(p.max()) <= 1


def test_order_index():
    assert order_index(0.05, 64) == 3            # Proposition 1: floor(eps (S+1))
    assert order_index(0.05, 64, delta=0.1) == 1  # Proposition 2 (PAC)
    assert order_index(0.05, 499) == 25


def test_proposition1_outage():
    """Empirical outage of the safety layer must be close to m/(S+1) when the aging model is correct."""
    snr, h, rho = gen_instances(40, 10, seed=1)
    s, hh, r = map(jnp.array, (snr, h, rho))
    p = jnp.array(wmmse(snr * np.abs(h) ** 2))
    rate = quantile_rate(sample_true_gains(jax.random.PRNGKey(2), s, hh, r, 64), p, 0.05)
    G = sample_true_gains(jax.random.PRNGKey(3), s, hh, r, 2000)
    C = jnp.log2(1 + sinr_from_gains(G, p[:, None, :]))
    act = np.array(rate) > 1e-6
    out = np.array((C < rate[:, None, :]).mean(1))[act].mean()
    assert abs(out - 3 / 65) < 0.02, out


def test_conformal_quantile():
    """Proposition 3 on synthetic exchangeable scores: P(s_test < c_hat) <= eps."""
    rng = np.random.default_rng(0); eps, N = 0.05, 200; hits = []
    for _ in range(2000):
        sc = rng.lognormal(size=N + 1); c = np.sort(sc[:N])[int(np.floor(eps * (N + 1))) - 1]
        hits.append(sc[N] < c)
    assert np.mean(hits) <= eps + 0.01


if __name__ == '__main__':
    for f in [test_shapes, test_order_index, test_proposition1_outage, test_conformal_quantile]:
        f(); print('passed:', f.__name__)
