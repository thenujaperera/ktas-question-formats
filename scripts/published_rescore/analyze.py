import pandas as pd, numpy as np, sys
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
d=pd.read_csv(sys.argv[1])
d['tsig']=(d.dtau_lo>0)|(d.dtau_hi<0); d['Dsig']=(d.dD_lo>0)|(d.dD_hi<0)
d['confirm']=d.tsig&d.Dsig&~d.disagree; d['opp']=d.tsig&d.Dsig&d.disagree
d['unconf']=d.tsig&~d.confirm
d['Dmissed']=d.Dsig&~d.confirm
d['revD']=d.disagree&d.Dsig
g=d.groupby('study').agg(n=('disagree','size'),dis=('disagree','sum'),tsig=('tsig','sum'),confirm=('confirm','sum'),unconf=('unconf','sum'),opp=('opp','sum'),revD=('revD','sum'),Dsig=('Dsig','sum'),accdis=('acc_disagree_D','sum'),adt=('dties',lambda x: np.mean(np.abs(x))))
g.loc['TOTAL']=g.sum(); g.loc['TOTAL','adt']=d.dties.abs().mean(); print(g)
dis=d[d.disagree]
print('acc* with D in disagreements',(np.sign(dis.dacc)==np.sign(dis.dD)).sum(),'of',len(dis),'; with tau',(np.sign(dis.dacc)==np.sign(dis.dtau)).sum())
print('tau favours more-tied',((dis.tiesA>dis.tiesB)==(dis.dtau>0)).sum())
d['bin']=pd.cut(d.dties.abs(),[0,0.1,0.2,0.3,1],include_lowest=True)
print(d.groupby('bin',observed=True).agg(n=('disagree','size'),dis=('disagree','sum'),revD=('revD','sum'),unconf=('unconf','sum'),tsig=('tsig','sum')))
print(d[d.opp|d.revD][['study','aspect','A','B','dtau','dtau_lo','dtau_hi','dD','dD_lo','dD_hi','accA','accB','tiesA','tiesB']].round(3).to_string())
