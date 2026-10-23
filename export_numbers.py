"""Export every number quoted in the paper text and tables to N.json (needs results.pkl and bench.pkl)."""
import pickle, json, numpy as np
from scipy.stats import wilcoxon
R=pickle.load(open('results.pkl','rb')); Bn=pickle.load(open('bench.pkl','rb'))
f=lambda x,n=2: f"{float(x):.{n}f}"
N={}; M=R['main']
N['ag_gp']=f(M['AgeGuard']['gp']); N['ag_out']=f(M['AgeGuard']['out'],3); N['wn_out']=f(M['WMMSE-Naive']['out'],2)
pct=lambda a,b: f(100*(a-b)/b,1)
N['g_wn']=pct(M['AgeGuard']['gp'],M['WMMSE-Naive']['gp']); N['g_wq']=pct(M['AgeGuard']['gp'],M['WMMSE-Q']['gp'])
N['g_ew']=pct(M['AgeGuard']['gp'],M['EWMMSE-Q']['gp']); N['g_pc']=pct(M['AgeGuard']['gp'],M['GNN-PerfCSI-Q']['gp']); N['g_qo']=pct(M['AgeGuard']['gp'],M['Q-Opt']['gp'])
b=Bn[20]; N['lat_ag']=f(b['gnn']+b['safety']); N['lat_wq']=f(b['wmmse']+b['safety']); N['lat_qo']=f(b['qopt']+b['safety'],1)
N['speed']=f((b['wmmse']+b['safety'])/(b['gnn']+b['safety']),1); N['speed_q']=f((b['qopt']+b['safety'])/(b['gnn']+b['safety']),0)
order=['FP-Naive','WMMSE-Naive','FP-Q','WMMSE-Q','EWMMSE-Q','GNN-PerfCSI-Q','Q-Opt','AgeGuard']
N['tab_main']=[[m,f(M[m]['gp'])+' ± '+f(M[m]['gp_std']),f(M[m]['out'],3),f(M[m]['p95'],3),f(M[m]['viol'],2),f(M[m]['meanP'],2),f(M[m]['active'],2)] for m in order]
P=R['pac']; N['tab_pac']=[[m,f(M[m]['gp']),f(M[m]['viol'],3),f(P[m]['gp']),f(P[m]['out'],4),f(P[m]['viol'],3)] for m in ['FP-Q','WMMSE-Q','EWMMSE-Q','AgeGuard']]
N['pac_ag_gp']=f(P['AgeGuard']['gp']); N['pac_ag_viol']=f(P['AgeGuard']['viol'],3); N['pac_wq_gp']=f(P['WMMSE-Q']['gp']); N['m_viol']=f(M['AgeGuard']['viol'],2)
g={m:M[m]['link_gp'].reshape(200,-1).sum(1) for m in M}; st=[]
for bm in ['WMMSE-Q','EWMMSE-Q','GNN-PerfCSI-Q','Q-Opt']:
    d=g['AgeGuard']-g[bm]; st.append([bm,f(d.mean()),'['+f(d.mean()-1.96*d.std(ddof=1)/np.sqrt(len(d)))+', '+f(d.mean()+1.96*d.std(ddof=1)/np.sqrt(len(d)))+']',f(100*(d>0).mean(),1)+'%',f"{wilcoxon(d).pvalue:.1e}"])
N['tab_stat']=st
V=R['vsweep']; N['tab_v']=[[f(v,0)]+[f(V[v][m]['gp'],1) for m in ['WMMSE-Naive','WMMSE-Q','EWMMSE-Q','GNN-PerfCSI-Q','AgeGuard']]+[pct(V[v]['AgeGuard']['gp'],V[v]['GNN-PerfCSI-Q']['gp'])+'%'] for v in sorted(V)]
N['v60_pc']=pct(V[60.0]['AgeGuard']['gp'],V[60.0]['GNN-PerfCSI-Q']['gp']); N['v60_wq']=pct(V[60.0]['AgeGuard']['gp'],V[60.0]['WMMSE-Q']['gp']); N['v0_pc']=pct(V[0.0]['AgeGuard']['gp'],V[0.0]['GNN-PerfCSI-Q']['gp'])
N['wn_out_range']=f(min(V[v]['WMMSE-Naive']['out'] for v in V if v>0),2)+'–'+f(max(V[v]['WMMSE-Naive']['out'] for v in V if v>0),2)
K=R['ksweep']; N['tab_k']=[[str(k)]+[f(K[k][m]['gp'],1) for m in ['WMMSE-Q','EWMMSE-Q','GNN-PerfCSI-Q','Q-Opt','AgeGuard']]+[f(K[k]['AgeGuard']['out'],3)] for k in sorted(K)]
N['k40_wq']=pct(K[40]['AgeGuard']['gp'],K[40]['WMMSE-Q']['gp']); N['k10_wq']=pct(K[10]['AgeGuard']['gp'],K[10]['WMMSE-Q']['gp'])
E=R['esweep']; N['tab_e']=[[f(e)]+[f(E[e][m]['gp'],1) for m in ['WMMSE-Q','EWMMSE-Q','GNN-PerfCSI-Q','AgeGuard']]+[f(E[e]['AgeGuard']['out'],4)] for e in sorted(E)]
N['e01_wq']=pct(E[0.01]['AgeGuard']['gp'],E[0.01]['WMMSE-Q']['gp']); N['e10_wq']=pct(E[0.1]['AgeGuard']['gp'],E[0.1]['WMMSE-Q']['gp'])
S=R['ssweep']; from core import order_index
N['tab_s']=[[str(s),str(order_index(0.05,s)),f(order_index(0.05,s)/(s+1),4)]+[f(S[s][m]['gp'],1) for m in ['WMMSE-Q','AgeGuard']]+[f(S[s]['AgeGuard']['out'],4)] for s in sorted(S)]
X=R['mis']; N['tab_mis']=[[f(k,1)]+[f(X[k][m]['gp'],1)+' / '+f(X[k][m]['out'],3) for m in ['WMMSE-Q','EWMMSE-Q','AgeGuard']] for k in sorted(X)]
N['mis2_out']=f(X[2.0]['AgeGuard']['out'],3); N['mis05_out']=f(X[0.5]['AgeGuard']['out'],4)
A=R['abl']; N['tab_abl']=[[k,f(v[0]),f(v[1],3),pct(v[0],A['Full AgeGuard'][0])+'%'] for k,v in A.items()]
N['tab_bench']=[[str(k),f(Bn[k]['gnn']),f(Bn[k]['safety']),f(Bn[k]['gnn']+Bn[k]['safety']),f(Bn[k]['wmmse']),f(Bn[k]['wmmse']+Bn[k]['safety']),f(Bn[k]['qopt']+Bn[k]['safety'],1)] for k in sorted(Bn)]
json.dump(N,open('N.json','w'),indent=1); print({k:v for k,v in N.items() if not k.startswith('tab')})
