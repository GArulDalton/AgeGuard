"""Draw Figs. 1-9 from results.pkl."""
import pickle, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family':'serif','font.size':10,'axes.grid':True,'grid.alpha':0.3,'savefig.dpi':220})
R=pickle.load(open('results.pkl','rb'))
ST={'FP-Naive':('grey',':','x'),'WMMSE-Naive':('tab:orange',':','s'),'FP-Q':('tab:brown','--','v'),
    'WMMSE-Q':('tab:blue','--','o'),'EWMMSE-Q':('tab:purple','--','D'),'GNN-PerfCSI-Q':('tab:olive','-.','^'),
    'Q-Opt':('k','-','*'),'AgeGuard':('tab:red','-','P')}
def lines(ax,xs,D,key,methods,ylabel,xlabel):
    for m in methods:
        c,ls,mk=ST[m]; ax.plot(xs,[D[x][m][key] for x in xs],color=c,ls=ls,marker=mk,label=m,lw=1.6 if m!='AgeGuard' else 2.4)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
# Fig 1: system model
fig,ax=plt.subplots(figsize=(6.6,3.3)); ax.set_xlim(0,10); ax.set_ylim(0,5); ax.axis('off')
rng=np.random.default_rng(3)
pos=[(1.0,3.6),(3.2,1.2),(5.0,3.7),(1.6,1.0),(4.0,2.6)]
for i,(x,y) in enumerate(pos):
    ax.plot(x,y,'^',ms=11,color='tab:blue'); ax.plot(x+0.7,y-0.35,'o',ms=9,color='tab:green')
    ax.annotate('',(x+0.62,y-0.3),(x+0.08,y-0.04),arrowprops=dict(arrowstyle='->',lw=1.8,color='tab:blue'))
    ax.text(x-0.25,y+0.2,f'Tx{i+1}',fontsize=8); ax.text(x+0.72,y-0.72,f'Rx{i+1}',fontsize=8)
for (a,b) in [(0,4),(2,4),(4,1),(3,1),(1,4)]:
    x,y=pos[a]; X,Y=pos[b]
    ax.annotate('',(X+0.62,Y-0.3),(x+0.1,y),arrowprops=dict(arrowstyle='->',lw=1,ls='--',color='tab:red',alpha=0.7))
ax.text(0.4,4.6,'Dense D2D / interference network: desired (blue), interference (red, dashed)',fontsize=8.5)
ax.add_patch(FancyBboxPatch((6.4,0.4),3.4,4.1,boxstyle='round,pad=0.05',fc='#f4f4f4',ec='k'))
ax.text(6.55,4.15,'Per-receiver CSI aging',fontsize=9,weight='bold')
ax.annotate('',(9.6,3.5),(6.6,3.5),arrowprops=dict(arrowstyle='->'))
ax.text(9.2,3.2,'time',fontsize=8)
ax.plot([6.9,6.9],[3.3,3.7],'k'); ax.text(6.6,3.75,r'estimate $\hat{h}$',fontsize=8)
ax.plot([8.6,8.6],[3.3,3.7],'k'); ax.text(8.2,3.75,r'transmit $h$',fontsize=8)
ax.annotate('',(8.6,3.0),(6.9,3.0),arrowprops=dict(arrowstyle='<->',color='tab:red')); ax.text(7.45,2.65,r'delay $\tau_k$',fontsize=8,color='tab:red')
ax.text(6.55,2.0,r'$h=\rho_k\hat{h}+\sqrt{1-\rho_k^2}\,e$',fontsize=9.5)
ax.text(6.55,1.4,r'$\rho_k=J_0(2\pi f_{D,k}\tau_k)$',fontsize=9.5)
ax.text(6.55,0.75,r'$\Rightarrow$ rate/power must hedge',fontsize=8.5)
plt.tight_layout(); plt.savefig('fig1_system.png'); plt.close()
# Fig 2: architecture
fig,ax=plt.subplots(figsize=(6.8,2.3)); ax.set_xlim(0,14); ax.set_ylim(-0.5,3); ax.axis('off')
blocks=[(0.1,'Aged CSI\n'+r'$\{\hat h_{jk}\},\{\rho_k\}$','#dfe9f7'),(2.45,'Age-aware\nnode/edge\nfeatures','#e8f4e0'),
        (4.8,'T-layer message\npassing GNN\n(mean+max agg.)','#fde9d9'),(7.15,'Power head\n'+r'$p_k=\sigma(\cdot)P_{\max}$','#fdf2cc'),
        (9.5,'Quantile safety\nlayer: S aged\nsamples, $C_{(m)}$','#f6d6d6'),(11.85,'Output\n'+r'$(p_k, r_k)$'+'\nguaranteed\noutage','#e4e4e4')]
for x,t,c in blocks:
    ax.add_patch(FancyBboxPatch((x,0.6),2.0,1.8,boxstyle='round,pad=0.05',fc=c,ec='k')); ax.text(x+1.0,1.5,t,ha='center',va='center',fontsize=7.0)
for x,_,_ in blocks[:-1]: ax.annotate('',(x+2.4,1.5),(x+2.08,1.5),arrowprops=dict(arrowstyle='->',lw=1.4))
ax.annotate('',(5.8,0.55),(10.5,0.55),arrowprops=dict(arrowstyle='->',ls='--',color='tab:red',connectionstyle='arc3,rad=-0.25'))
ax.text(5.2,-0.35,r'training: $\nabla_\theta \sum_k C_{k,(m)}$ back-propagated through order statistic',fontsize=7.5,color='tab:red')
plt.tight_layout(); plt.savefig('fig2_arch.png'); plt.close()
# Fig 3: convergence
fig,ax=plt.subplots(figsize=(4.8,3.2))
for k,lab,c in [('hist_main','AgeGuard (T=3)','tab:red'),('hist_nofeat','w/o age features','tab:green'),('hist_T1','AgeGuard (T=1)','tab:blue')]:
    h=np.array(R[k]); y=np.convolve(h[:,1],np.ones(10)/10,'valid'); ax.plot(h[9:,0],y,color=c,label=lab)
ax.set_xlabel('Training iteration'); ax.set_ylabel('Safe sum rate (bps/Hz)'); ax.legend(); plt.tight_layout(); plt.savefig('fig3_conv.png'); plt.close()
meths=['FP-Naive','WMMSE-Naive','WMMSE-Q','EWMMSE-Q','GNN-PerfCSI-Q','AgeGuard']
# Fig 4: Doppler sweep goodput and outage
xs=sorted(R['vsweep']); fig,axs=plt.subplots(1,2,figsize=(7.2,3.1))
lines(axs[0],xs,R['vsweep'],'gp',meths,'Avg. sum goodput (bps/Hz)','Max. user speed (km/h)')
lines(axs[1],xs,R['vsweep'],'out',meths,'Avg. outage probability','Max. user speed (km/h)')
axs[1].axhline(0.05,color='k',lw=0.8); axs[1].set_ylim(-0.02,0.6)
axs[0].legend(fontsize=7); plt.tight_layout(); plt.savefig('fig4_doppler.png'); plt.close()
# Fig 5: K sweep
xs=sorted(R['ksweep']); fig,ax=plt.subplots(figsize=(4.8,3.3))
lines(ax,xs,R['ksweep'],'gp',['WMMSE-Naive','WMMSE-Q','EWMMSE-Q','Q-Opt','AgeGuard'],'Avg. sum goodput (bps/Hz)','Number of links K')
ax.legend(fontsize=7.5); plt.tight_layout(); plt.savefig('fig5_K.png'); plt.close()
# Fig 6: eps sweep
xs=sorted(R['esweep']); fig,axs=plt.subplots(1,2,figsize=(7.2,3.1))
lines(axs[0],xs,R['esweep'],'gp',['WMMSE-Q','EWMMSE-Q','GNN-PerfCSI-Q','AgeGuard'],'Avg. sum goodput (bps/Hz)',r'Target outage $\varepsilon$')
lines(axs[1],xs,R['esweep'],'out',['WMMSE-Q','EWMMSE-Q','GNN-PerfCSI-Q','AgeGuard'],'Measured outage',r'Target outage $\varepsilon$')
axs[1].plot(xs,xs,'k:',label='target'); 
for a in axs: a.set_xscale('log')
axs[0].legend(fontsize=7); plt.tight_layout(); plt.savefig('fig6_eps.png'); plt.close()
# Fig 7: CDF per-link goodput
fig,ax=plt.subplots(figsize=(4.8,3.2))
for m in ['WMMSE-Naive','WMMSE-Q','EWMMSE-Q','Q-Opt','AgeGuard']:
    v=np.sort(R['main'][m]['link_gp']); c,ls,_=ST[m]; ax.plot(v,np.arange(1,len(v)+1)/len(v),color=c,ls=ls,label=m)
ax.set_xlabel('Per-link goodput (bps/Hz)'); ax.set_ylabel('CDF'); ax.legend(fontsize=7.5); plt.tight_layout(); plt.savefig('fig7_cdf.png'); plt.close()
# Fig 8: per-link outage histogram marginal vs PAC
fig,ax=plt.subplots(figsize=(4.8,3.2)); bins=np.linspace(0,0.25,41)
ax.hist(R['main']['AgeGuard']['link_out'],bins,alpha=0.6,label=r'AgeGuard, marginal ($m=3$)',color='tab:red',density=True)
ax.hist(R['pac']['AgeGuard']['link_out'],bins,alpha=0.6,label=r'AgeGuard-PAC ($\delta=0.1$, $m=1$)',color='tab:blue',density=True)
ax.axvline(0.05,color='k',ls='--'); ax.set_xlabel('Per-link outage probability'); ax.set_ylabel('Density'); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig('fig8_pac.png'); plt.close()
# Fig 9: mismatch
xs=sorted(R['mis']); fig,axs=plt.subplots(1,2,figsize=(7.2,3.1))
lines(axs[0],xs,R['mis'],'gp',['WMMSE-Q','EWMMSE-Q','AgeGuard'],'Avg. sum goodput (bps/Hz)',r'Doppler mismatch factor $\kappa$')
lines(axs[1],xs,R['mis'],'out',['WMMSE-Q','EWMMSE-Q','AgeGuard'],'Measured outage',r'Doppler mismatch factor $\kappa$')
axs[1].axhline(0.05,color='k',lw=0.8); axs[0].legend(fontsize=7.5); plt.tight_layout(); plt.savefig('fig9_mis.png'); plt.close()
print('figs ok')
