const API = '/api';
let charts = {};
let currentModel = 'GRU';

const $ = (id) => document.getElementById(id);
const fmt = (v, n=2) => Number(v).toFixed(n);

async function getJSON(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`API error ${res.status}`);
    return res.json();
}

function setText(id, value) {
    const el = $(id);
    if (el) el.textContent = value;
}

async function loadStatus() {
    const d = await getJSON(`${API}/status`);
    setText('systemStatus', 'System Online');
    setText('systemSub', `${d.models_loaded} models loaded`);
    setText('datasetRows', d.dataset_rows.toLocaleString());
}

async function loadSensors() {
    const d = await getJSON(`${API}/sensors`);
    const s = d.sensors;
    setText('currentPM25', fmt(s['PM2.5']));
    setText('temperature', fmt(s.Temperature, 1));
    setText('humidity', fmt(s.Humidity, 1));
    setText('sensorPM10', `${fmt(s.PM10)} µg/m³`);
    setText('sensorCO', `${fmt(s.CO, 2)} ppm`);
    setText('sensorNO', `${fmt(s.NO, 2)} ppb`);
    setText('sensorO3', `${fmt(s.O3, 2)} ppb`);
    setText('sensorSO2', `${fmt(s.SO2, 2)} ppb`);
    setText('sensorTVOC', `${fmt(s.TVOC, 2)} ppm`);
    setText('sensorWind', `${fmt(s['Wind Speed'], 2)} m/s`);
    setText('sensorHumidity', `${fmt(s.Humidity, 1)} %RH`);
    setText('sensorDirection', `${fmt(s['Wind Direction'], 0)}°`);
    setText('lastUpdate', d.datetime);
}

async function loadForecast() {
    const d = await getJSON(`${API}/forecast?model=${encodeURIComponent(currentModel)}`);
    setText('forecastPM25', fmt(d.prediction));
    setText('forecastTime', d.forecast_datetime);
    setText('forecastModel', d.model);
    const change = d.prediction - d.current_pm25;
    setText('predictionChange', `${change >= 0 ? '+' : ''}${fmt(change)} µg/m³ vs current`);
}

async function loadModels() {
    const d = await getJSON(`${API}/models`);
    const tbody = $('modelTableBody');
    if (!tbody) return;
    tbody.innerHTML = '';
    d.models.forEach((m) => {
        const tr = document.createElement('tr');
        if (m.rank === 1) tr.className = 'best-row';
        tr.innerHTML = `<td><span class="rank">${m.rank}</span></td><td><strong>${m.model}</strong></td><td>${fmt(m.MAE,3)}</td><td>${fmt(m.RMSE,3)}</td><td class="r2">${fmt(m.R2,3)}</td>`;
        tbody.appendChild(tr);
    });
    setText('bestModelName', d.best_model.model);
    setText('bestMAE', fmt(d.best_model.MAE,3));
    setText('bestRMSE', fmt(d.best_model.RMSE,3));
    setText('bestR2', fmt(d.best_model.R2,3));
}

async function loadPredictions() {
    const d = await getJSON(`${API}/predictions?limit=100`);
    const rows = d.rows;
    const labels = rows.map(r => new Date(r.datetime).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}));
    const actual = rows.map(r => r.actual);
    const predicted = rows.map(r => r[currentModel]);
    makeChart('trendChart', 'line', labels, [
        {label:'Actual PM2.5', data:actual, borderWidth:2, pointRadius:0, tension:.25},
        {label:`${currentModel} Prediction`, data:predicted, borderWidth:2, pointRadius:0, tension:.25}
    ], 'PM2.5 (µg/m³)');

    const scatter = rows.filter(r => r.actual != null && r[currentModel] != null).slice(-80);
    makeChart('scatterChart', 'scatter', [], [{label:'Actual vs Predicted', data:scatter.map(r=>({x:r.actual,y:r[currentModel]})), pointRadius:4}], 'Prediction');

    const residual = scatter.map(r=>r.actual-r[currentModel]);
    makeChart('residualChart', 'line', scatter.map((_,i)=>i+1), [{label:'Residual',data:residual,borderWidth:1.5,pointRadius:0,tension:.15}], 'Residual (µg/m³)');
}

async function loadXAI() {
    const d = await getJSON(`${API}/xai`);
    const labels=d.global.map(x=>x.feature.replace('PM25/ug/m3','PM2.5'));
    const values=d.global.map(x=>x.importance_pct);
    makeChart('xaiChart','bar',labels,[{label:'Mean |SHAP| (%)',data:values,borderWidth:0}], 'Importance (%)', true);
    const box=$('xaiList');
    if(box){ box.innerHTML=''; d.local.slice(0,6).forEach(x=>{ const div=document.createElement('div'); div.className=`explanation-item ${x.shap>=0?'positive':'negative'}`; div.innerHTML=`<span>${x.feature.replace('PM25/ug/m3','PM2.5')}</span><strong>${x.shap>=0?'+':''}${fmt(x.shap,3)}</strong>`; box.appendChild(div); }); }
}

function makeChart(id, type, labels, datasets, yTitle, isHorizontal=false) {
    const el=$(id); if(!el || typeof Chart==='undefined') return;
    if(charts[id]) charts[id].destroy();
    const options={responsive:true,maintainAspectRatio:false,plugins:{legend:{display:true,position:'top'}},scales:{x:{title:{display:!!labels.length,text:''}},y:{title:{display:true,text:yTitle}}}};
    if(type==='scatter') options.scales={x:{title:{display:true,text:'Actual PM2.5'}},y:{title:{display:true,text:'Predicted PM2.5'}}};
    if(isHorizontal){ options.indexAxis='y'; options.scales={x:{title:{display:true,text:yTitle}},y:{ticks:{autoSkip:false}}}; }
    charts[id]=new Chart(el,{type,data:{labels,datasets},options});
}

async function refreshDashboard() {
    try {
        await Promise.all([loadStatus(),loadSensors(),loadForecast(),loadModels(),loadPredictions(),loadXAI()]);
        setText('refreshStatus','Updated just now');
    } catch(err) {
        console.error(err);
        setText('systemStatus','Backend Offline');
        setText('systemSub','Run: python app.py');
    }
}

document.addEventListener('DOMContentLoaded',()=>{
    const selector=$('modelSelect');
    if(selector) selector.addEventListener('change',async e=>{currentModel=e.target.value; await loadForecast(); await loadPredictions();});
    const refresh=$('refreshBtn');
    if(refresh) refresh.addEventListener('click',refreshDashboard);
    document.querySelectorAll('.nav-item').forEach(a=>a.addEventListener('click',e=>{
        document.querySelectorAll('.nav-item').forEach(x=>x.classList.remove('active')); e.currentTarget.classList.add('active');
    }));
    refreshDashboard();
});
