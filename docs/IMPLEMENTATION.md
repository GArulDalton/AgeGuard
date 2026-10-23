# AgeGuard implementation details

This document explains how the code implements each part of the paper, the array conventions, every public function, and how to extend the method.

## 1. Pipeline overview

```
positions, speeds, delays ──► gen_instances ──► (snr, h_est, rho)         Sec. III
                                   │
                                   ▼
                       feats (age-aware features)                         Eqs. (6)-(7)
                                   │
                                   ▼
                gnn_forward (T message-passing layers) ──► powers p       Eqs. (8)-(9)
                                   │
                                   ▼
   sample_true_gains (S aged samples) ──► quantile_rate ──► rates r       Eq. (10), Props. 1-2
                                   │
                    (optional) conformal back-off c_hat ──► c_hat · r     Eqs. (12)-(13), Prop. 3
                    (optional) online update from ACK/NACK                Eq. (14), Prop. 4
```

Training (`train.py`) runs the same forward path and maximizes the sum of the rates from the safety layer by back-propagating through the sort.

## 2. Array conventions

| Name | Shape | Meaning |
|---|---|---|
| `snr` | `[B, K, K]` | `snr[b, j, k]` = P_max · β_jk / σ²: large-scale gain from transmitter j to receiver k, normalized by noise |
| `h` / `h_est` | `[B, K, K]` complex | aged small-scale fading estimates ĥ_jk (unit variance) |
| `rho` | `[B, K]` | aging coefficient of receiver k, ρ_k = J0(2π f_D,k τ_k) |
| `p` | `[B, K]` | transmit power as a fraction of P_max, in [0, 1] |
| `G`, `Gs` | `[B, S, K, K]` | channel gain samples, P_max · β_jk · \|h_jk\|² / σ² |
| rates `r` | `[B, K]` | bps/Hz |

B is the number of networks (batch), K the number of links and S the number of safety-layer samples. The diagonal `[k, k]` holds the desired links.

## 3. Module reference

### `core.py`

| Function | Purpose | Paper |
|---|---|---|
| `P` | Default system parameters: 500 m area, 10–50 m links, 3.5 GHz, 20 dBm, −95 dBm noise, 0.5–2 ms delay, 30 km/h max speed | Table 1 |
| `gen_instances(B, K, seed, vmax_kmh)` | Draws positions, path loss (35.3 + 37.6 log10 d), Rayleigh estimates, speeds and delays. Returns `snr, h, rho`. Also stores `gen_instances.last_x = 2π f_D τ`, used to build mismatched ρ_true = J0(κ·x). | Sec. III-A/B |
| `sample_true_gains(key, snr, h_est, rho, S)` | Draws S channels from the aging model h = ρ ĥ + √(1−ρ²) e, applying ρ of the receiver. | Eq. (2) |
| `sinr_from_gains(G, p)` | SINR of every link; interference uses a masked sum to avoid floating-point cancellation. | Eq. (1) |
| `order_index(eps, S, delta=None)` | m = ⌊ε(S+1)⌋ (Prop. 1), or with `delta` the PAC index from the Beta distribution (Prop. 2). | Props. 1–2 |
| `quantile_rate(Gs, p, eps, delta=None)` | Safety layer: sorts the S sampled capacities per link and returns the m-th smallest. Differentiable with respect to p. | Eq. (10) |
| `feats(snr, h_est, rho, age_aware=True)` | Node features [ĝ_kk, ḡ_kk, ρ_k, ρ_k²] and edge features [ĝ_jk, ĝ_kj, ḡ_jk, ρ_k, ρ_j], in dB/10. With `age_aware=False` the aging inputs are zeroed (ablation and perfect-CSI baseline). | Eqs. (6)–(7) |
| `init_gnn(key, H=64, T=3)` | Initializes the input MLP, T message and update MLPs, and the power head (zero-initialized so training starts at p = 0.5). | Sec. IV-A |
| `gnn_forward(params, node, edge)` | Message passing with mean + max aggregation and residual updates; sigmoid power head. | Eqs. (8)–(9) |
| `wmmse(G, iters=100)` | Classical WMMSE for the SISO interference channel. | baseline [1] |

### `train.py`

| Function | Purpose |
|---|---|
| `make_loss(eps, S, age_aware)` | Loss = −mean sum of safety-layer rates, using fresh aged samples each call (Eq. 11). |
| `train(K, steps, B, S, eps, age_aware, vmax, seed, H, T)` | Adam with cosine-decayed learning rate. Every iteration uses B new random networks (no fixed dataset). Returns `(params, history)`. |

### `evalu.py`

| Function | Purpose |
|---|---|
| `eval_true(snr, h, rho, p, r, key, St)` | Draws St fresh true channels (never used for decisions) and returns per-link goodput r·1{C ≥ r} and outage. |
| `qrate_np(...)` | Batched NumPy wrapper around `quantile_rate`. |
| `est_gain`, `mean_gain` | Gains from aged estimates, and conditional mean gains ḡ (used by EWMMSE). |
| `qopt(...)` | Q-Opt baseline: per-instance Adam on the empirical quantile objective (300 steps). |

### `recalibration.py`

| Function | Purpose | Paper |
|---|---|---|
| `scenario(B, seed, kappa, hetero)` | Test networks whose true aging uses κ times the assumed Doppler, or per-receiver κ_k ~ U[0.5, 3]. | Sec. 6.9 |
| `cal_scores(...)` | One random active link per calibration network; score = observed capacity / model-based rate. | Eq. (12) |
| `conformal_c(scores, eps)` | ĉ = ⌊ε(N+1)⌋-th smallest score. | Eq. (13), Prop. 3 |
| `prepare`, `evaluate` | Model-based rates A and true-capacity draws; outage and goodput for any factor c. | Table 8 |
| main block, online section | Log-multiplier update θ ← θ + η(ε − NACK rate) over 900 intervals with κ = 1 → 2 → 0.5. | Eq. (14), Alg. 3, Fig. 13 |

## 4. Training details

- **Data:** synthetic, generated on the fly. Each iteration draws 32 new networks of K = 20 links with speeds up to 60 km/h, so the model sees the whole range of aging coefficients.
- **Objective:** maximize Σ_k C_k,(m) with m = 3 (S = 64, ε = 0.05). Because every link's outage is fixed at m/(S+1) by the safety layer, this equals maximizing goodput.
- **Gradient:** JAX differentiates through `jnp.sort`. The gradient flows to the sample at rank m, and the reparameterized noise makes the estimator unbiased almost everywhere.
- **Optimizer:** Adam, learning rate 1e-3 with cosine decay to 5e-5 over 3,000 iterations.
- **Architecture:** hidden size 64, 3 message-passing layers, GELU activations, 96,449 parameters.
- **Seeds:** model seed 0 for AgeGuard; data seeds 10,000 + iteration.

## 5. Reproducibility

- All randomness comes from explicit NumPy seeds and JAX PRNG keys.
- `pretrained/ageguard_main.pkl` is the AgeGuard model used in the paper, stored as NumPy arrays (`params`), with its training curve (`history`) and configuration.
- `results/recal.pkl` holds the recalibration results behind Tables 8–9 and Figs. 12–13.
- `train_models.py` and `evaluate.py` regenerate everything else; see the README for the full pipeline.

Load the pretrained model:

```python
import pickle, jax, jax.numpy as jnp
d = pickle.load(open('pretrained/ageguard_main.pkl', 'rb'))
params = jax.tree_util.tree_map(jnp.asarray, d['params'])
```

## 6. Extending AgeGuard

- **Another channel or aging model:** replace `sample_true_gains`. The safety layer and Proposition 1 only need i.i.d. samples from the model you believe; recalibration (Prop. 3) needs only observed outcomes.
- **Another objective** (e.g., proportional fairness): change the loss in `train.make_loss`, for example to Σ log(r_k + ε₀). The outage guarantee is unaffected because it comes from the safety layer.
- **Multi-antenna links:** replace the scalar gains with effective post-beamforming gains in `sinr_from_gains`, and add beamformer outputs to the GNN head.
- **Larger networks:** keep only the strongest neighbours of each link as edges in `gnn_forward` to reduce the cost from O(K²) to O(K).
