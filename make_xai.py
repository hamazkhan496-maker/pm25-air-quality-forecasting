import pandas as pd, numpy as np, joblib, json
from pathlib import Path
import shap
BASE=Path('/mnt/data'); ART=BASE/'model_artifacts'
df=pd.read_csv(BASE/'air_quality.csv'); df['datetime']=pd.to_datetime(df['Date'].astype(str)+' '+df['Time'].astype(str),errors='coerce'); df=df.sort_values('datetime').reset_index(drop=True)
TARGET='PM25/ug/m3';
for c in df.columns:
 if c not in ['Date','Time','datetime']: df[c]=pd.to_numeric(df[c],errors='coerce')
work=df.copy(); work['target_future']=work[TARGET].shift(-1)
for lag in [1,2,3,6,12,24,48]: work[f'{TARGET}_lag_{lag}']=work[TARGET].shift(lag)
for win in [3,6,12,24]:
 work[f'{TARGET}_rollmean_{win}']=work[TARGET].rolling(win).mean(); work[f'{TARGET}_rollstd_{win}']=work[TARGET].rolling(win).std()
work['hour']=work['datetime'].dt.hour+work['datetime'].dt.minute/60; work['hour_sin']=np.sin(2*np.pi*work['hour']/24); work['hour_cos']=np.cos(2*np.pi*work['hour']/24); work['dow']=work['datetime'].dt.dayofweek; work['dow_sin']=np.sin(2*np.pi*work['dow']/7); work['dow_cos']=np.cos(2*np.pi*work['dow']/7); work['wind_dir_sin']=np.sin(np.deg2rad(work['wind_degree/Ўг'])); work['wind_dir_cos']=np.cos(np.deg2rad(work['wind_degree/Ўг']))
work=work.replace([np.inf,-np.inf],np.nan).dropna(subset=['target_future']).reset_index(drop=True)
feats=joblib.load(ART/'feature_columns.joblib'); imp=joblib.load(ART/'tabular_imputer.joblib'); sc=joblib.load(ART/'tabular_scaler.joblib'); model=joblib.load(ART/'extra_trees.joblib')
X=sc.transform(imp.transform(work[feats]));
# use a deterministic test slice
n=len(work); ntr=int(n*.70); nv=int(n*.15); Xt=X[ntr+nv:]
idx=np.linspace(0,len(Xt)-1,min(250,len(Xt)),dtype=int); sample=Xt[idx]
expl=shap.TreeExplainer(model); vals=expl.shap_values(sample); vals=np.asarray(vals)
mean_abs=np.mean(np.abs(vals),axis=0)
out=pd.DataFrame({'feature':feats,'mean_abs_shap':mean_abs}).sort_values('mean_abs_shap',ascending=False).head(15)
out['importance_pct']=100*out.mean_abs_shap/out.mean_abs_shap.sum(); out.to_csv(ART/'shap_global.csv',index=False)
# local explanation for latest prediction
latest=X[-1:]; lv=np.asarray(expl.shap_values(latest)).reshape(-1); base=float(np.asarray(expl.expected_value).reshape(-1)[0]); pred=float(model.predict(latest)[0])
local=pd.DataFrame({'feature':feats,'shap':lv,'abs_shap':np.abs(lv)}).sort_values('abs_shap',ascending=False).head(12); local['direction']=np.where(local['shap']>=0,'increases','decreases'); local.to_csv(ART/'shap_local.csv',index=False)
with open(ART/'xai_meta.json','w') as f: json.dump({'model':'Extra Trees','method':'SHAP TreeExplainer','base_value':base,'prediction':pred},f,indent=2)
print(out.to_string(index=False)); print('local saved')
