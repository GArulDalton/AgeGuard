"""Draw Figs. 10-11 from recal.pkl."""
import pickle, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.size':10,'axes.grid':True,'grid.alpha':0.3,'savefig.dpi':220})
R=pickle.load(open('recal.pkl','rb')); S=R['sweep']; ks=[0.5,1.0,1.5,2.0,2.5,3.0]
fig,ax=plt.subplots(2,1,figsize=(4.8,4.6),sharex=True)
for meth,c in [('WMMSE','tab:blue'),('AgeGuard','tab:red')]:
    ax[0].plot(ks,[S[k][meth]['model'][0] for k in ks],'--o',color=c,label=f'{meth}-Q (model only)',ms=4)
    ax[0].plot(ks,[S[k][meth]['rc'][0] for k in ks],'-s',color=c,label=f'{meth}-RC (recalibrated)',ms=4,lw=2 if meth=='AgeGuard' else 1.4)
    ax[1].plot(ks,[S[k][meth]['model'][1] for k in ks],'--o',color=c,ms=4)
    ax[1].plot(ks,[S[k][meth]['rc'][1] for k in ks],'-s',color=c,ms=4,lw=2 if meth=='AgeGuard' else 1.4)
ax[0].axhline(0.05,color='k',lw=0.8); ax[0].set_ylabel('Measured outage'); ax[0].legend(fontsize=7)
ax[1].set_xlabel(r'Doppler mismatch factor $\kappa$'); ax[1].set_ylabel('Avg. sum goodput (bps/Hz)')
plt.tight_layout(); plt.savefig('fig10_recal.png'); plt.close()
A=R['aci']; T=len(A['model']); w=50
roll=lambda v: np.convolve(v,np.ones(w)/w,'valid')
fig,ax=plt.subplots(2,1,figsize=(4.8,4.4),sharex=True)
t=np.arange(w-1,T)
ax[0].plot(t,roll(A['model']),color='grey',ls='--',label='AgeGuard (model only)')
ax[0].plot(t,roll(A['aci']),color='tab:red',label='AgeGuard-OR (online)')
ax[0].axhline(0.05,color='k',lw=0.8); ax[0].set_ylabel('Outage (50-slot avg.)'); ax[0].legend(fontsize=7.5)
for x in [300,600]:
    for a in ax: a.axvline(x,color='k',ls=':',lw=0.8)
ax[0].text(120,0.17,r'$\kappa=1$',fontsize=8); ax[0].text(420,0.17,r'$\kappa=2$',fontsize=8); ax[0].text(720,0.17,r'$\kappa=0.5$',fontsize=8)
ax[1].plot(np.arange(T),A['c'],color='tab:red'); ax[1].set_ylabel(r'Rate multiplier $c_t$'); ax[1].set_xlabel('Scheduling interval t')
plt.tight_layout(); plt.savefig('fig11_online.png'); plt.close(); print('ok')
