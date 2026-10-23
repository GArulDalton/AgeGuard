"""Train the four GNN models used in the paper (AgeGuard, perfect-CSI GNN, no-age-feature and 1-layer ablations).
Output: models.pkl  (about 25 min on one CPU core). Resumable."""
import pickle, os
from train import *
M = pickle.load(open('models.pkl','rb')) if os.path.exists('models.pkl') else {}
cfg=[('main',dict(vmax=60.0,seed=0)),('noaoi',dict(vmax=0.0,age_aware=False,seed=1)),
     ('nofeat',dict(vmax=60.0,age_aware=False,seed=2)),('T1',dict(vmax=60.0,T=1,seed=3))]
for n,c in cfg:
    if n in M: continue
    t=time.time(); p,h=train(steps=3000,log=False,**c); M[n]=(p,h)
    pickle.dump(M,open('models.pkl','wb')); print(n,time.time()-t,flush=True)
