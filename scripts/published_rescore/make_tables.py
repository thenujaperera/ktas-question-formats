import pandas as pd, numpy as np, sys
d=pd.read_csv(sys.argv[1]); out=sys.argv[2]
d['tsig']=(d.dtau_lo>0)|(d.dtau_hi<0); d['Dsig']=(d.dD_lo>0)|(d.dD_hi<0)
d['confirm']=d.tsig&d.Dsig&~d.disagree; d['opp']=d.tsig&d.Dsig&d.disagree
d['unc_ns']=d.tsig&~d.Dsig; d['revD']=d.disagree&d.Dsig; d['gap']=d.dties.abs()
f=lambda x:f"{x:+.3f}".replace('-','−')
ci=lambda a,lo,hi:f"{f(a)} [{f(lo)}, {f(hi)}]"
order=['G-Eval (proxy)','Wang et al.','Shen et al.','Chiang & Lee SummEval','Chiang & Lee Topical-Chat']
names={'G-Eval (proxy)':'G-Eval (proxies)','Chiang & Lee SummEval':'Chiang & Lee, SummEval','Chiang & Lee Topical-Chat':'Chiang & Lee, Topical-Chat'}
def row(name,x):
    return f"| {name} | {len(x)} | {int(x.disagree.sum())} | {int(x.tsig.sum())} | {int(x.confirm.sum())} | {int(x.unc_ns.sum())} | {int(x.opp.sum())} | {int(x.revD.sum())} | {x.gap.mean():.2f} |"
T=["| Study | n | Disagree | Sig. τ-b | Confirmed | d n.s. | d opposite | Sig. d, τ-b opposite | Tie gap |","|---|---|---|---|---|---|---|---|---|"]
for s in order: T.append(row(names.get(s,s),d[d.study==s]))
nc=d[~d.study.str.startswith('Chiang')]
T.append(row('Without Chiang & Lee',nc)); T.append(row('All studies',d))
tables={'TABLE_COUNTS':"\n".join(T)}
d['bin']=pd.cut(d.gap,[0,0.1,0.2,0.3,1],include_lowest=True,labels=['≤ 0.1','0.1–0.2','0.2–0.3','> 0.3'])
G=["| Tie gap | Comparisons | τ-b and d disagree | Significant τ-b | Not confirmed by d |","|---|---|---|---|---|"]
for b,x in d.groupby('bin',observed=True): G.append(f"| {b} | {len(x)} | {int(x.disagree.sum())} ({100*x.disagree.mean():.0f}%) | {int(x.tsig.sum())} | {int((x.tsig&~x.confirm).sum())} |")
tables['TABLE_GAP']="\n".join(G)
S=["| Aspect | τ-b GPT-3.5 / GPT-4 | Δτ-b [95% CI] | Δd [95% CI] | Δacc* [95% CI] | Tied pairs GPT-3.5 / GPT-4 |","|---|---|---|---|---|---|"]
for _,r in d[(d.study=='Shen et al.')&(d.A=='ChatGPT rts')&(d.B=='GPT-4 rts')].iterrows():
    S.append(f"| {r.aspect.capitalize()} | {r.tauA:.3f} / {r.tauB:.3f} | {ci(r.dtau,r.dtau_lo,r.dtau_hi)} | {ci(r.dD,r.dD_lo,r.dD_hi)} | {ci(r.dacc,r.dacc_lo,r.dacc_hi)} | {100*r.tiesA:.0f}% / {100*r.tiesB:.0f}% |")
tables['TABLE_SHEN']="\n".join(S)
W=["| Aspect | Baseline | τ-b judge / baseline | Δτ-b [95% CI] | Δd [95% CI] | Δacc* [95% CI] |","|---|---|---|---|---|---|"]
bn={'bart_score_cnn_src_hypo':'BARTScore-CNN','bart_score_para_src_hypo':'BARTScore-CNN-Para'}
w=d[(d.study=='Wang et al.')&d.B.isin(bn)]
for asp in ['coherence','consistency','fluency','relevance']:
    for _,r in w[w.aspect==asp].iterrows():
        W.append(f"| {asp.capitalize()} | {bn[r.B]} | {r.tauA:.3f} / {r.tauB:.3f} | {ci(r.dtau,r.dtau_lo,r.dtau_hi)} | {ci(r.dD,r.dD_lo,r.dD_hi)} | {ci(r.dacc,r.dacc_lo,r.dacc_hi)} |")
tables['TABLE_WANG']="\n".join(W)
E=["| Aspect | Single rating | τ-b level / mean | Δτ-b [95% CI] | Δd [95% CI] | Δacc* [95% CI] | Tied pairs (level) |","|---|---|---|---|---|---|---|"]
for _,r in d[d.study=='G-Eval (proxy)'].iterrows():
    E.append(f"| {r.aspect.capitalize()} | {'Mode of 20' if r.A=='mode' else 'First sample'} | {r.tauA:.3f} / {r.tauB:.3f} | {ci(r.dtau,r.dtau_lo,r.dtau_hi)} | {ci(r.dD,r.dD_lo,r.dD_hi)} | {ci(r.dacc,r.dacc_lo,r.dacc_hi)} | {100*r.tiesA:.0f}% |")
tables['TABLE_GEVAL']="\n".join(E)
dis=d[d.disagree]; asig=(dis.dacc_lo>0)|(dis.dacc_hi<0)
withD=asig&(np.sign(dis.dacc)==np.sign(dis.dD)); withT=asig&(np.sign(dis.dacc)==np.sign(dis.dtau))
allsig=(d.dacc_lo>0)|(d.dacc_hi<0)
tables['ACC_SIG_SENTENCE']=(f"With intervals, acc\\* was significant in {int(asig.sum())} of the 33 disagreements: on the side of Somers' D in {int(withD.sum())} and on the side of τ-b in {int(withT.sum())}. "
  f"Over all 204 comparisons, acc\\* disagreed in sign with Somers' D in {int((np.sign(d.dacc)!=np.sign(d.dD)).sum())}, mostly in Wang et al., where it tuned a tie threshold for the continuous metrics.")
import json; json.dump(tables,open(out,'w'),ensure_ascii=False,indent=1)
print(tables['TABLE_COUNTS']); print(tables['ACC_SIG_SENTENCE'])
print('acc* with D among disagreements (sign):',int((np.sign(dis.dacc)==np.sign(dis.dD)).sum()))
# figure
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['pdf.fonttype']=42; matplotlib.rcParams['ps.fonttype']=42; import matplotlib.pyplot as plt
sg=np.where(d.tiesA>=d.tiesB,1,-1); x=d.dD*sg; y=d.dtau*sg
fig,ax=plt.subplots(figsize=(3.4,3.0))
sc=ax.scatter(x,y,c=d.gap,cmap='viridis_r',s=12,edgecolors='k',linewidths=0.2)
ax.axhline(0,color='0.5',lw=0.6); ax.axvline(0,color='0.5',lw=0.6)
lim=max(abs(x).max(),abs(y).max())*1.05; ax.plot([-lim,lim],[-lim,lim],':',color='0.6',lw=0.6)
ax.set_xlabel('Δd (more-tied minus less-tied)',fontsize=7); ax.set_ylabel('Δτ-b (more-tied minus less-tied)',fontsize=7)
ax.tick_params(labelsize=6); cb=fig.colorbar(sc,ax=ax); cb.set_label('Tie gap',fontsize=7); cb.ax.tick_params(labelsize=6)
ax.text(-lim*0.95,lim*0.85,'τ-b favours more-tied,\nd favours the other',fontsize=6,va='top')
fig.tight_layout(); fig.savefig(sys.argv[3]); print('fig ok')
