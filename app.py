from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import torch
import torch.nn as nn

BASE = Path(__file__).resolve().parent
ART = BASE / 'model_artifacts'
CSV = BASE / 'air_quality.csv'
TARGET = 'PM25/ug/m3'

app = FastAPI(title='PM2.5 One-Step Forecasting API', version='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

# ---------------- Neural network definitions ----------------
class RNNModel(nn.Module):
    def __init__(self, d, h=48):
        super().__init__(); self.r=nn.RNN(d,h,batch_first=True); self.f=nn.Sequential(nn.Linear(h,24),nn.ReLU(),nn.Linear(24,1))
    def forward(self,x):
        o,_=self.r(x); return self.f(o[:,-1]).squeeze(-1)
class LSTMModel(nn.Module):
    def __init__(self,d,h=48):
        super().__init__(); self.r=nn.LSTM(d,h,batch_first=True); self.f=nn.Sequential(nn.Linear(h,24),nn.ReLU(),nn.Linear(24,1))
    def forward(self,x):
        o,_=self.r(x); return self.f(o[:,-1]).squeeze(-1)
class GRUModel(nn.Module):
    def __init__(self,d,h=48):
        super().__init__(); self.r=nn.GRU(d,h,batch_first=True); self.f=nn.Sequential(nn.Linear(h,24),nn.ReLU(),nn.Linear(24,1))
    def forward(self,x):
        o,_=self.r(x); return self.f(o[:,-1]).squeeze(-1)
class CNNModel(nn.Module):
    def __init__(self,d):
        super().__init__(); self.c=nn.Sequential(nn.Conv1d(d,48,3,padding=1),nn.ReLU(),nn.Conv1d(48,24,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool1d(1)); self.f=nn.Linear(24,1)
    def forward(self,x): return self.f(self.c(x.transpose(1,2)).squeeze(-1)).squeeze(-1)
class CNNLSTMModel(nn.Module):
    def __init__(self,d):
        super().__init__(); self.c=nn.Sequential(nn.Conv1d(d,48,3,padding=1),nn.ReLU(),nn.Conv1d(48,24,3,padding=1),nn.ReLU()); self.r=nn.LSTM(24,48,batch_first=True); self.f=nn.Sequential(nn.Linear(48,24),nn.ReLU(),nn.Linear(24,1))
    def forward(self,x):
        z=self.c(x.transpose(1,2)).transpose(1,2); o,_=self.r(z); return self.f(o[:,-1]).squeeze(-1)

# ---------------- Load data/artifacts ----------------
df = pd.read_csv(CSV)
df['datetime'] = pd.to_datetime(df['Date'].astype(str)+' '+df['Time'].astype(str), errors='coerce')
df = df.sort_values('datetime').reset_index(drop=True)
NUM_COLS = [c for c in df.columns if c not in ['Date','Time','datetime']]
for c in NUM_COLS: df[c] = pd.to_numeric(df[c], errors='coerce')

metrics = json.load(open(ART/'metrics.json', encoding='utf-8'))
FEATURES = joblib.load(ART/'feature_columns.joblib')
TAB_IMP = joblib.load(ART/'tabular_imputer.joblib')
TAB_SC = joblib.load(ART/'tabular_scaler.joblib')
SEQ_IMP = joblib.load(ART/'sequence_imputer.joblib')
SEQ_SC = joblib.load(ART/'sequence_scaler.joblib')
SEQ_FEATURES = joblib.load(ART/'sequence_features.joblib')

CLASSICAL = ['Ridge','Lasso','Linear Regression','Gradient Boosting','Random Forest','XGBoost','Extra Trees','KNN','SVR','MLP','Decision Tree']
DEEP = ['RNN','LSTM','GRU','CNN','CNN-LSTM']
ALL_MODELS = CLASSICAL + DEEP
FILE_MAP = {
    'Ridge':'ridge.joblib','Lasso':'lasso.joblib','Linear Regression':'linear_regression.joblib',
    'Gradient Boosting':'gradient_boosting.joblib','Random Forest':'random_forest.joblib','XGBoost':'xgboost.joblib',
    'Extra Trees':'extra_trees.joblib','KNN':'knn.joblib','SVR':'svr.joblib','MLP':'mlp.joblib','Decision Tree':'decision_tree.joblib'
}
loaded_classical = {name: joblib.load(ART/FILE_MAP[name]) for name in CLASSICAL}

def deep_model(name):
    d=len(SEQ_FEATURES)
    m={'RNN':RNNModel(d),'LSTM':LSTMModel(d),'GRU':GRUModel(d),'CNN':CNNModel(d),'CNN-LSTM':CNNLSTMModel(d)}[name]
    m.load_state_dict(torch.load(ART/(name.lower().replace('-','_')+'.pt'), map_location='cpu'))
    m.eval(); return m
loaded_deep = {name: deep_model(name) for name in DEEP}

# ---------------- Feature engineering ----------------
def make_features(data):
    w=data.copy()
    for lag in [1,2,3,6,12,24,48]: w[f'{TARGET}_lag_{lag}']=w[TARGET].shift(lag)
    for win in [3,6,12,24]:
        w[f'{TARGET}_rollmean_{win}']=w[TARGET].rolling(win).mean()
        w[f'{TARGET}_rollstd_{win}']=w[TARGET].rolling(win).std()
    w['hour']=w['datetime'].dt.hour+w['datetime'].dt.minute/60
    w['hour_sin']=np.sin(2*np.pi*w['hour']/24); w['hour_cos']=np.cos(2*np.pi*w['hour']/24)
    w['dow']=w['datetime'].dt.dayofweek; w['dow_sin']=np.sin(2*np.pi*w['dow']/7); w['dow_cos']=np.cos(2*np.pi*w['dow']/7)
    w['wind_dir_sin']=np.sin(np.deg2rad(w['wind_degree/Ўг'])); w['wind_dir_cos']=np.cos(np.deg2rad(w['wind_degree/Ўг']))
    return w.replace([np.inf,-np.inf],np.nan)

WORK = make_features(df)

def tabular_input_latest():
    row=WORK.iloc[[-1]][FEATURES]
    return TAB_SC.transform(TAB_IMP.transform(row))

def sequence_input_latest():
    raw=df[SEQ_FEATURES].replace([np.inf,-np.inf],np.nan)
    arr=SEQ_SC.transform(SEQ_IMP.transform(raw))
    return np.asarray(arr[-12:],dtype=np.float32)[None,:,:]

def predict_model(name):
    if name in CLASSICAL:
        return float(loaded_classical[name].predict(tabular_input_latest())[0])
    with torch.no_grad(): return float(loaded_deep[name](torch.tensor(sequence_input_latest())).numpy()[0])

# ---------------- API ----------------
@app.get('/api/status')
def status():
    return {'status':'online','dataset_rows':int(len(df)),'models_loaded':len(ALL_MODELS),'target':TARGET,'forecast_type':'one-step-ahead','interval_minutes':int(round(df['datetime'].diff().dt.total_seconds().dropna().median()/60))}

@app.get('/api/models')
def models():
    ranking=[]
    for name in ALL_MODELS:
        m=metrics['metrics'][name]['test']
        ranking.append({'model':name,'MAE':round(m['MAE'],4),'RMSE':round(m['RMSE'],4),'R2':round(m['R2'],4)})
    ranking.sort(key=lambda x:x['RMSE'])
    for i,r in enumerate(ranking,1): r['rank']=i
    return {'models':ranking,'best_model':ranking[0]}

@app.get('/api/forecast')
def forecast(model: str = Query('GRU')):
    if model not in ALL_MODELS: model='GRU'
    value=predict_model(model)
    last=df.iloc[-1]
    step=df['datetime'].diff().dt.total_seconds().dropna().median()
    next_time=last['datetime']+pd.to_timedelta(step,unit='s')
    all_preds={name:predict_model(name) for name in ALL_MODELS}
    return {'model':model,'prediction':value,'current_pm25':float(last[TARGET]),'current_datetime':str(last['datetime']),'forecast_datetime':str(next_time),'all_predictions':all_preds}

@app.get('/api/sensors')
def sensors():
    r=df.iloc[-1]
    return {'datetime':str(r['datetime']),'sensors':{
        'PM2.5':float(r[TARGET]),'PM10':float(r['PM10/ug/m3']),'CO':float(r['CO/ppm']),'NO':float(r['NO/ppb']),
        'O3':float(r['O3/ppb']),'SO2':float(r['SO2/ppb']),'Temperature':float(r['Temperature']),
        'TVOC':float(r['tvoc/ppm']),'Wind Speed':float(r['wind_speed/m/s']),'Humidity':float(r['Humidity/%RH']),
        'Wind Direction':float(r['wind_degree/Ўг'])}}

@app.get('/api/predictions')
def predictions(limit:int=100):
    p=pd.read_csv(ART/'test_predictions.csv').tail(max(10,min(limit,500)))
    return {'rows':p.where(pd.notnull(p),None).to_dict(orient='records')}

@app.get('/api/xai')
def xai():
    g=pd.read_csv(ART/'shap_global.csv').head(15)
    l=pd.read_csv(ART/'shap_local.csv').head(12)
    return {'method':'SHAP TreeExplainer','model':'XGBoost','global':g.to_dict(orient='records'),'local':l.to_dict(orient='records')}

@app.get('/')
def root(): return FileResponse(BASE/'index.html')

app.mount('/static',StaticFiles(directory=BASE),name='static')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=8000)
