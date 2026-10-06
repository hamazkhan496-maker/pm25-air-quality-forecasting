import os, json, warnings, joblib
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BASE=Path('/mnt/data')
ART=BASE/'model_artifacts'; ART.mkdir(exist_ok=True)
CSV=BASE/'air_quality.csv'
SEED=42
np.random.seed(SEED); torch.manual_seed(SEED)

# ---------- data ----------
df=pd.read_csv(CSV)
df['datetime']=pd.to_datetime(df['Date'].astype(str)+' '+df['Time'].astype(str), errors='coerce')
df=df.sort_values('datetime').reset_index(drop=True)
TARGET='PM25/ug/m3'
num_cols=[c for c in df.columns if c not in ['Date','Time','datetime']]
for c in num_cols: df[c]=pd.to_numeric(df[c],errors='coerce')
# one-step target
df['target_future']=df[TARGET].shift(-1)
# engineered tabular features at T
work=df.copy()
for lag in [1,2,3,6,12,24,48]:
    work[f'{TARGET}_lag_{lag}']=work[TARGET].shift(lag)
for win in [3,6,12,24]:
    work[f'{TARGET}_rollmean_{win}']=work[TARGET].rolling(win).mean()
    work[f'{TARGET}_rollstd_{win}']=work[TARGET].rolling(win).std()
# cyclical time features
work['hour']=work['datetime'].dt.hour + work['datetime'].dt.minute/60
work['hour_sin']=np.sin(2*np.pi*work['hour']/24)
work['hour_cos']=np.cos(2*np.pi*work['hour']/24)
work['dow']=work['datetime'].dt.dayofweek
work['dow_sin']=np.sin(2*np.pi*work['dow']/7)
work['dow_cos']=np.cos(2*np.pi*work['dow']/7)
# wind direction cyclic
wd='wind_degree/Ўг'
work['wind_dir_sin']=np.sin(np.deg2rad(work[wd]))
work['wind_dir_cos']=np.cos(np.deg2rad(work[wd]))

feature_cols=[c for c in work.columns if c not in ['Date','Time','datetime','target_future','hour','dow']]
work=work.replace([np.inf,-np.inf],np.nan)
# chronological split on rows after feature creation
valid=work['target_future'].notna()
work=work.loc[valid].reset_index(drop=True)
n=len(work); ntr=int(n*.70); nval=int(n*.15)
train=work.iloc[:ntr]; val=work.iloc[ntr:ntr+nval]; test=work.iloc[ntr+nval:]
Xtr=train[feature_cols].values; Xv=val[feature_cols].values; Xte=test[feature_cols].values
ytr=train['target_future'].values; yv=val['target_future'].values; yte=test['target_future'].values
imp=SimpleImputer(strategy='median'); sc=StandardScaler()
Xtr_i=imp.fit_transform(Xtr); Xv_i=imp.transform(Xv); Xte_i=imp.transform(Xte)
Xtr_s=sc.fit_transform(Xtr_i); Xv_s=sc.transform(Xv_i); Xte_s=sc.transform(Xte_i)
joblib.dump(imp,ART/'tabular_imputer.joblib'); joblib.dump(sc,ART/'tabular_scaler.joblib')
joblib.dump(feature_cols,ART/'feature_columns.joblib')

metrics={}; preds={}

def eval_model(name, model, X1=Xtr_s, X2=Xv_s, X3=Xte_s):
    print('Training',name,flush=True); model.fit(X1,ytr)
    pv=model.predict(X2); pt=model.predict(X3)
    metrics[name]={
      'val': {'MAE':float(mean_absolute_error(yv,pv)),'RMSE':float(np.sqrt(mean_squared_error(yv,pv))),'R2':float(r2_score(yv,pv))},
      'test': {'MAE':float(mean_absolute_error(yte,pt)),'RMSE':float(np.sqrt(mean_squared_error(yte,pt))),'R2':float(r2_score(yte,pt))}
    }
    preds[name]=pt
    joblib.dump(model,ART/f'{name.lower().replace(" ","_").replace("-","_")}.joblib')

models={
'Ridge':Ridge(alpha=1.0),
'Lasso':Lasso(alpha=0.001,max_iter=5000),
'Linear Regression':LinearRegression(),
'Gradient Boosting':GradientBoostingRegressor(n_estimators=250,max_depth=3,learning_rate=.05,random_state=SEED),
'Random Forest':RandomForestRegressor(n_estimators=250,max_depth=None,min_samples_leaf=2,n_jobs=-1,random_state=SEED),
'XGBoost':XGBRegressor(n_estimators=300,max_depth=5,learning_rate=.04,subsample=.85,colsample_bytree=.85,objective='reg:squarederror',n_jobs=-1,random_state=SEED),
'Extra Trees':ExtraTreesRegressor(n_estimators=250,min_samples_leaf=1,n_jobs=-1,random_state=SEED),
'KNN':KNeighborsRegressor(n_neighbors=10,weights='distance',p=2,n_jobs=-1),
'SVR':SVR(C=10,epsilon=.05,gamma='scale'),
'MLP':MLPRegressor(hidden_layer_sizes=(128,64,32),activation='relu',solver='adam',alpha=1e-4,learning_rate_init=.001,max_iter=500,early_stopping=True,validation_fraction=.15,random_state=SEED),
'Decision Tree':DecisionTreeRegressor(max_depth=18,min_samples_leaf=2,random_state=SEED)
}
for name,m in models.items(): eval_model(name,m)

# ---------- sequence data for DL ----------
raw_features=num_cols
raw=df[raw_features].copy().replace([np.inf,-np.inf],np.nan)
raw_imp=SimpleImputer(strategy='median'); raw_sc=StandardScaler()
# fit on training portion only, then transform all
raw_tr=raw.iloc[:ntr]; raw_imp.fit(raw_tr); raw_i=raw_imp.transform(raw); raw_s=raw_sc.fit_transform(raw_i) if False else None
# correct scaler fit on train-imputed
raw_tr_i=raw_imp.transform(raw.iloc[:ntr]); raw_sc.fit(raw_tr_i); raw_s=raw_sc.transform(raw_imp.transform(raw))
joblib.dump(raw_imp,ART/'sequence_imputer.joblib'); joblib.dump(raw_sc,ART/'sequence_scaler.joblib'); joblib.dump(raw_features,ART/'sequence_features.joblib')
seq_len=12
# sequence sample index i predicts target at i+1, sequence ends at i
Xseq=[]; yseq=[]; idxs=[]
for i in range(seq_len-1,len(df)-1):
    Xseq.append(raw_s[i-seq_len+1:i+1]); yseq.append(df[TARGET].iloc[i+1]); idxs.append(i+1)
Xseq=np.asarray(Xseq,dtype=np.float32); yseq=np.asarray(yseq,dtype=np.float32); idxs=np.asarray(idxs)
# split by target row boundaries
mask_tr=idxs < ntr; mask_val=(idxs>=ntr)&(idxs<ntr+nval); mask_te=idxs>=ntr+nval
Xseq_tr,Xseq_v,Xseq_te=Xseq[mask_tr],Xseq[mask_val],Xseq[mask_te]
yseq_tr,yseq_v,yseq_te=yseq[mask_tr],yseq[mask_val],yseq[mask_te]

class RNNModel(nn.Module):
    def __init__(self,d,h=64): super().__init__(); self.r=nn.RNN(d,h,batch_first=True); self.fc=nn.Sequential(nn.Linear(h,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x): o,_=self.r(x); return self.fc(o[:,-1]).squeeze(-1)
class LSTMModel(nn.Module):
    def __init__(self,d,h=64): super().__init__(); self.r=nn.LSTM(d,h,batch_first=True); self.fc=nn.Sequential(nn.Linear(h,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x): o,_=self.r(x); return self.fc(o[:,-1]).squeeze(-1)
class GRUModel(nn.Module):
    def __init__(self,d,h=64): super().__init__(); self.r=nn.GRU(d,h,batch_first=True); self.fc=nn.Sequential(nn.Linear(h,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x): o,_=self.r(x); return self.fc(o[:,-1]).squeeze(-1)
class CNNModel(nn.Module):
    def __init__(self,d):
        super().__init__(); self.c=nn.Sequential(nn.Conv1d(d,64,3,padding=1),nn.ReLU(),nn.Conv1d(64,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool1d(1)); self.fc=nn.Linear(32,1)
    def forward(self,x): return self.fc(self.c(x.transpose(1,2)).squeeze(-1)).squeeze(-1)
class CNNLSTMModel(nn.Module):
    def __init__(self,d):
        super().__init__(); self.c=nn.Sequential(nn.Conv1d(d,64,3,padding=1),nn.ReLU(),nn.Conv1d(64,32,3,padding=1),nn.ReLU()); self.r=nn.LSTM(32,64,batch_first=True); self.fc=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x): z=self.c(x.transpose(1,2)).transpose(1,2); o,_=self.r(z); return self.fc(o[:,-1]).squeeze(-1)

class MLPSeq(nn.Module):
    def __init__(self,d,seq): super().__init__(); self.fc=nn.Sequential(nn.Flatten(),nn.Linear(d*seq,128),nn.ReLU(),nn.Dropout(.1),nn.Linear(64 if False else 128,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,x): return self.fc(x).squeeze(-1)

def train_torch(name,model,epochs=12):
    print('Training',name,flush=True); device=torch.device('cpu'); model.to(device)
    ds=TensorDataset(torch.tensor(Xseq_tr),torch.tensor(yseq_tr)); dl=DataLoader(ds,batch_size=64,shuffle=True)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3); lossfn=nn.MSELoss(); best=None; bestv=np.inf; patience=3; bad=0
    xv=torch.tensor(Xseq_v); yvv=torch.tensor(yseq_v)
    for ep in range(epochs):
        model.train()
        for xb,yb in dl:
            opt.zero_grad(); loss=lossfn(model(xb),yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl=float(lossfn(model(xv),yvv))
        if vl<bestv-1e-5:
            bestv=vl; best={k:v.detach().clone() for k,v in model.state_dict().items()}; bad=0
        else: bad+=1
        if bad>=patience: break
    model.load_state_dict(best); model.eval()
    with torch.no_grad(): pv=model(xv).numpy(); pt=model(torch.tensor(Xseq_te)).numpy()
    metrics[name]={'val': {'MAE':float(mean_absolute_error(yseq_v,pv)),'RMSE':float(np.sqrt(mean_squared_error(yseq_v,pv))),'R2':float(r2_score(yseq_v,pv))},'test': {'MAE':float(mean_absolute_error(yseq_te,pt)),'RMSE':float(np.sqrt(mean_squared_error(yseq_te,pt))),'R2':float(r2_score(yseq_te,pt))}}
    preds[name]=pt
    torch.save(model.state_dict(),ART/f'{name.lower().replace(" ","_").replace("-","_")}.pt')
    return model

D=Xseq.shape[2]
deep_models={
'RNN':RNNModel(D), 'LSTM':LSTMModel(D), 'GRU':GRUModel(D), 'CNN':CNNModel(D), 'CNN-LSTM':CNNLSTMModel(D)
}
trained_deep={}
for name,m in deep_models.items(): trained_deep[name]=train_torch(name,m,epochs=12)
# PyTorch MLP is not used because sklearn MLP already covers MLP.

# test timestamps/predictions for dashboard
# align test target timestamps
te_times=df['datetime'].iloc[ntr+nval:].astype(str).tolist()
# save test predictions for all models (some classical use same test target rows; deep test starts at ntr+nval but sequence begins later; should align)
pred_df=pd.DataFrame({'datetime':te_times,'actual':yte})
for name,pt in preds.items():
    if len(pt)==len(pred_df): pred_df[name]=pt
    else:
        # deep arrays should be one per test target row except first sequence offset; fill NaN at beginning if needed
        arr=np.full(len(pred_df),np.nan); arr[-len(pt):]=pt; pred_df[name]=arr
pred_df.to_csv(ART/'test_predictions.csv',index=False)

# model ranking
ranking=[]
for name,m in metrics.items():
    ranking.append({'model':name,**m['test']})
ranking=sorted(ranking,key=lambda x:x['RMSE'])
with open(ART/'metrics.json','w') as f: json.dump({'metrics':metrics,'ranking':ranking,'target':TARGET,'sequence_length':seq_len,'feature_count':len(feature_cols),'rows':len(df),'split':{'train':ntr,'validation':nval,'test':len(test)}},f,indent=2)

# simple feature importance from Extra Trees; fallback for best model if needed
et=joblib.load(ART/'extra_trees.joblib')
imp_df=pd.DataFrame({'feature':feature_cols,'importance':et.feature_importances_}).sort_values('importance',ascending=False).head(15)
imp_df.to_csv(ART/'feature_importance.csv',index=False)

# latest row info
latest=df.iloc[-1]
latest_payload={c:(None if pd.isna(latest[c]) else float(latest[c]) if c not in ['Date','Time'] else str(latest[c])) for c in num_cols}
latest_payload['datetime']=str(latest['datetime'])
with open(ART/'latest.json','w') as f: json.dump(latest_payload,f,indent=2)
print('\nDONE')
print(json.dumps(ranking[:5],indent=2))
