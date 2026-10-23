#!/usr/bin/env bash
# Reproduce every table and figure of the paper (about 1 hour on one CPU core).
set -e
python train_models.py        # -> models.pkl   (4 GNN models, ~25 min)
python evaluate.py            # -> results.pkl  (Tables 2-7, ~20 min)
python benchmark_latency.py   # -> bench.pkl    (Table 6)
python make_figures.py        # -> fig1..fig9 PNG files
python export_numbers.py      # -> N.json       (all numbers quoted in the text)
python recalibration.py       # -> recal.pkl    (Tables 8-9, ~6 min)
python make_recal_figures.py  # -> fig10_recal.png, fig11_online.png (paper Figs. 12-13)
python make_diagrams.py       # -> paper Figs. 3-4 (block diagrams)
echo "Done."
