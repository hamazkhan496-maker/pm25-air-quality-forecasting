import os,json,joblib,warnings
warnings.filterwarnings('ignore')
import numpy as np,pandas as pd,torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from pathlib import Path
BASE=Path('/mnt/data'); ART=BASE/'model_artifacts'; SEED=42
np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(2)
df=pd.read_csv(BASE/'air_quality.csv')
df['datetime']=pd.to_datetime(df['Date'].astype(str)+' '+df['Time'].astype(str),errors='coerce'); df=df.sort_values('datetime').reset_index(drop=True)
TARGET='PM25/ug/m3'; num_cols=[c for c in df.columns if c not in ['Date','Time','datetime']]
for c in num_cols: df[c]=pd.to_numeric(df[c],errors='coerce')
n=len(df); ntr=int((n-1)*.70); nval=int((n-1)*.15)
imp=joblib.load(ART/'sequence_imputer.joblib'); sc=joblib.load(ART/'sequence_scaler.joblib')
raw=df[num_cols].replace([np.inf,-np.inf],np.nan); raw_s=sc.transform(imp.transform(raw)); seq_len=12
X=[];y=[];idx=[]
for i in range(seq_len-1,len(df)-1): X.append(raw_s[i-seq_len+1:i+1]); y.append(df[TARGET].iloc[i+1]); idx.append(i+1)
X=np.asarray(X,np.float32); y=np.asarray(y,np.float32); idx=np.asarray(idx)
tr=idx<ntr; va=(idx>=ntr)&(idx<ntr+nval); te=idx>=ntr+nval
Xt,Xv,Xe=X[tr],X[va],X[te]; yt,yv,ye=y[tr],y[va],y[te]
class RNNModel(nn.Module):
 def __init__(self,d,h=48): super().__init__(); self.r=nn.RNN(d,h,batch_first=True); self.f=nn.Sequential(nn.Linear(h,24),nn.ReLU(),nn.Linear(24,1))
 def forward(self,x): o,_=self.r(x); return self.f(o[:,-1]).squeeze(-1)
class LSTMModel(nn.Module):
 def __init__(self,d,h=48): super().__init__(); self.r=nn.LSTM(d,h,batch_first=True); self.f=nn.Sequential(nn.Linear(h,24),nn.ReLU(),nn.Linear(24,1))
 def forward(self,x): o,_=self.r(x); return self.f(o[:,-1]).squeeze(-1)
class GRUModel(nn.Module):
 def __init__(self,d,h=48): super().__init__(); self.r=nn.GRU(d,h,batch_first=True); self.f=nn.Sequential(nn.Linear(h,24),nn.ReLU(),nn.Linear(24,1))
 def forward(self,x): o,_=self.r(x); return self.f(o[:,-1]).squeeze(-1)
class CNNModel(nn.Module):
 def __init__(self,d): super().__init__(); self.c=nn.Sequential(nn.Conv1d(d,48,3,padding=1),nn.ReLU(),nn.Conv1d(48,24,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool1d(1)); self.f=nn.Linear(24,1)
 def forward(self,x): return self.f(self.c(x.transpose(1,2)).squeeze(-1)).squeeze(-1)
class CNNLSTMModel(nn.Module):
 def __init__(self,d): super().__init__(); self.c=nn.Sequential(nn.Conv1d(d,48,3,padding=1),nn.ReLU(),nn.Conv1d(48,24,3,padding=1),nn.ReLU()); self.r=nn.LSTM(24,48,batch_first=True); self.f=nn.Sequential(nn.Linear(48,24),nn.ReLU(),nn.Linear(24,1))
 def forward(self,x): z=self.c(x.transpose(1,2)).transpose(1,2); o,_=self.r(z); return self.f(o[:,-1]).squeeze(-1)
models={'RNN':RNNModel(X.shape[2]),'LSTM':LSTMModel(X.shape[2]),'GRU':GRUModel(X.shape[2]),'CNN':CNNModel(X.shape[2]),'CNN-LSTM':CNNLSTMModel(X.shape[2])}
metrics={}; preds={}
for name,m in models.items():
 print('Training',name,flush=True); ds=TensorDataset(torch.tensor(Xt),torch.tensor(yt)); dl=DataLoader(ds,batch_size=128,shuffle=True); opt=torch.optim.Adam(m.parameters(),lr=.002); loss=nn.MSELoss(); best_state=None; best=1e99; bad=0
 xv=torch.tensor(Xv); yvt=torch.tensor(yv)
 for ep in range(6):
  m.train()
  for xb,yb in dl:
   opt.zero_grad(); l=loss(m(xb),yb); l.backward(); opt.step()
  m.eval()
  with torch.no_grad(): vl=float(loss(m(xv),yvt))
  print(' ',ep+1,vl,flush=True)
  if vl<best: best=vl; best_state={k:v.detach().clone() for k,v in m.state_dict().items()}; bad=0
  else: bad+=1
  if bad>=2: break
 m.load_state_dict(best_state); m.eval()
 with torch.no_grad(): pv=m(xv).numpy(); pe=m(torch.tensor(Xe)).numpy()
 metrics[name]={'val':{'MAE':float(mean_absolute_error(yv,pv)),'RMSE':float(np.sqrt(mean_squared_error(yv,pv))),'R2':float(r2_score(yv,pv))},'test':{'MAE':float(mean_absolute_error(ye,pe)),'RMSE':float(np.sqrt(mean_squared_error(ye,pe))),'R2':float(r2_score(ye,pe))}}
 preds[name]=pe; torch.save(m.state_dict(),ART/(name.lower().replace('-','_')+'.pt'))
# merge metrics from classical models by recomputing from saved predictions using test
# Load existing classical models and evaluate
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
imp2=joblib.load(ART/'tabular_imputer.joblib'); sc2=joblib.load(ART/'tabular_scaler.joblib'); feats=joblib.load(ART/'feature_columns.joblib')
work=df.copy(); work['target_future']=work[TARGET].shift(-1)
for lag in [1,2,3,6,12,24,48]: work[f'{TARGET}_lag_{lag}']=work[TARGET].shift(lag)
for win in [3,6,12,24]: work[f'{TARGET}_rollmean_{win}']=work[TARGET].rolling(win).mean(); work[f'{TARGET}_rollstd_{win}']=work[TARGET].rolling(win).std()
work['hour']=work['datetime'].dt.hour+work['datetime'].dt.minute/60; work['hour_sin']=np.sin(2*np.pi*work['hour']/24); work['hour_cos']=np.cos(2*np.pi*work['hour']/24); work['dow']=work['datetime'].dt.dayofweek; work['dow_sin']=np.sin(2*np.pi*work['dow']/7); work['dow_cos']=np.cos(2*np.pi*work['dow']/7); work['wind_dir_sin']=np.sin(np.deg2rad(work['wind_degree/Ўг'])); work['wind_dir_cos']=np.cos(np.deg2rad(work['wind_degree/Ўг']))
work=work.replace([np.inf,-np.inf],np.nan).dropna(subset=['target_future']).reset_index(drop=True)
Xall=sc2.transform(imp2.transform(work[feats])); yall=work['target_future'].values; Xtest=Xall[ntr+nval:]; ytest=yall[ntr+nval:]
for fn in ART.glob('*.joblib'):
 name=fn.stem.replace('_',' '); mapping={'linear regression':'Linear Regression','gradient boosting':'Gradient Boosting','random forest':'Random Forest','extra trees':'Extra Trees','decision tree':'Decision Tree','xgboost':'XGBoost','tabular imputer':'','tabular scaler':'','feature columns':'','sequence imputer':'','sequence scaler':'','sequence features':'','ridge':'Ridge','lasso':'Lasso','knn':'KNN','svr':'SVR','mlp':'MLP'}; name=mapping.get(name,name)
 if name in ['Ridge','Lasso','Linear Regression','Gradient Boosting','Random Forest','XGBoost','Extra Trees','KNN','SVR','MLP','Decision Tree']:
  try:
   mo=joblib.load(fn); pp=mo.predict(Xtest); metrics[name]={'test':{'MAE':float(mean_absolute_error(ytest,pp)),'RMSE':float(np.sqrt(mean_squared_error(ytest,pp))),'R2':float(r2_score(ytest,pp))}}
   preds[name]=pp
  except: pass
ranking=sorted([{'model':k,**v['test']} for k,v in metrics.items()],key=lambda z:z['RMSE'])
with open(ART/'metrics.json','w') as f: json.dump({'metrics':metrics,'ranking':ranking,'target':TARGET,'sequence_length':seq_len,'feature_count':len(feats),'rows':len(df),'split':{'train':ntr,'validation':nval,'test':len(df)-ntr-nval}},f,indent=2)
# aligned test predictions
out=pd.DataFrame({'datetime':df['datetime'].iloc[ntr+nval:].astype(str).values,'actual':df[TARGET].shift(-1).iloc[ntr+nval:].values})
for k,p in preds.items():
 if len(p)==len(out): out[k]=p
 elif len(p)<len(out):
  a=np.full(len(out),np.nan); a[-len(p):]=p; out[k]=a
out.to_csv(ART/'test_predictions.csv',index=False)
print('DONE ranking:'); print(json.dumps(ranking,indent=2))
