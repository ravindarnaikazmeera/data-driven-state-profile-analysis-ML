# server.py
import warnings
warnings.filterwarnings("ignore")

import io
import os
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ------------------ Configuration ------------------

# XLSX files live here (folder next to this file)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# All States & UTs (labels for UI)
# ---- All States & UTs (labels used in the dropdown) ----
ALL_REGIONS = {
    # States (28)
    "Andhra_Pradesh": "Andhra Pradesh",
    "Arunachal_Pradesh": "Arunachal Pradesh",
    "Assam": "Assam",
    "Bihar": "Bihar",
    "Chhattisgarh": "Chhattisgarh",
    "Goa": "Goa",
    "Gujarat": "Gujarat",
    "Haryana": "Haryana",
    "Himachal_Pradesh": "Himachal Pradesh",
    "Jharkhand": "Jharkhand",
    "Karnataka": "Karnataka",
    "Kerala": "Kerala",
    "Madhya_Pradesh": "Madhya Pradesh",
    "Maharashtra": "Maharashtra",
    "Manipur": "Manipur",
    "Meghalaya": "Meghalaya",
    "Mizoram": "Mizoram",
    "Nagaland": "Nagaland",
    "Odisha": "Odisha",
    "Punjab": "Punjab",
    "Rajasthan": "Rajasthan",
    "Sikkim": "Sikkim",
    "Tamil_Nadu": "Tamil Nadu",
    "Telangana": "Telangana",
    "Tripura": "Tripura",
    "Uttar_Pradesh": "Uttar Pradesh",
    "Uttarakhand": "Uttarakhand",
    "West_Bengal": "West Bengal",

    # Union Territories (8)
    "Andaman_Nicobar": "Andaman & Nicobar Islands",
    "Chandigarh": "Chandigarh",
    "Dadra_Nagar_Haveli_and_Daman_Diu": "Dadra & Nagar Haveli and Daman & Diu",
    "NCT_of_Delhi": "Delhi (NCT)",
    "Jammu_Kashmir": "Jammu & Kashmir",
    "Ladakh": "Ladakh",
    "Lakshadweep": "Lakshadweep",
    "Puducherry": "Puducherry",
}


# Available data files right now (you can add more later)
STATE_FILES = {
    "Andaman_Nicobar": "Andaman_Nicobar_latest10_clean_fix.xlsx",
    "Andhra_Pradesh": "Andhra_Pradesh_latest10_clean_fix.xlsx",
    "Arunachal_Pradesh": "Arunachal_Pradesh_latest10_clean_fix.xlsx",
    "Assam": "Assam_latest10_clean_fix.xlsx",
    "Bihar": "Bihar.xlsx",
    "Chhattisgarh": "Chhattisgarh.xlsx",
    "Goa": "Goa.xlsx",
    "Gujarat": "Gujarat.xlsx",
    "Haryana": "Haryana.xlsx",
    "Himachal_Pradesh": "Himachal_Pradesh.xlsx",
    "Jharkhand": "Jharkhand.xlsx",
    "Karnataka": "Karnataka.xlsx",
    "Kerala": "Kerala.xlsx",
    "Madhya_Pradesh": "Madhya_Pradesh.xlsx",
    "Maharashtra": "Maharashtra.xlsx",
    "Manipur": "Manipur.xlsx",
    "Meghalaya": "Meghalaya.xlsx",
    "Mizoram": "Mizoram.xlsx",
    "Nagaland": "Nagaland.xlsx",
    "Odisha": "Odisha.xlsx",
    "Punjab": "Punjab.xlsx",
    "Rajasthan": "Rajasthan.xlsx",
    "Sikkim": "Sikkim.xlsx",
    "Tamil_Nadu": "Tamil_Nadu.xlsx",
    "Telangana": "Telangana.xlsx",
    "Tripura": "Tripura.xlsx",
    "Uttar_Pradesh": "Uttar_Pradesh.xlsx",
    "Uttarakhand": "Uttarakhand.xlsx",
    "West_Bengal": "West_Bengal.xlsx",
    # ✅ Union Territories
    "Chandigarh": "Chandigarh.xlsx",
    "Dadra_Nagar_Haveli_and_Daman_Diu": "Dadra_Nagar_Haveli_and_Daman_Diu.xlsx",
    "NCT_of_Delhi": "NCT_of_Delhi.xlsx",
    "Jammu_Kashmir": "Jammu_Kashmir.xlsx",
    "Ladakh": "Ladakh.xlsx",
    "Lakshadweep": "Lakshadweep.xlsx",
    "Puducherry": "Puducherry.xlsx"
}
# ------------------ Rules / Actions ------------------

MASTER_ACTIONS = {
    "Share of agriculture in NSVA": [
        "Boost irrigation/input support; diversify into high-value crops/fisheries; expand agri-credit and insurance.",
        "Improve cold-chain and FPO linkages; train on climate-smart practices."
    ],
    "Cropping intensity": [
        "Scale micro-irrigation and water harvesting to enable Rabi; support short-duration varieties.",
        "Stabilize power for irrigation; repair canals; time-bound input subsidies."
    ],
    "Gross cropped area": [
        "Land development and fallow revival with MSP/contract farming certainty.",
        "Mechanization services (custom hiring) to ease labor constraints."
    ],
    "Air passenger traffic per million population": [
        "Promote tourism/connectivity; seasonal routes/fare support; last-mile access.",
        "Upgrade attractions/amenities; events in shoulder seasons; strong safety/health standards."
    ],
    "Electricity generation": [
        "Preventive maintenance and secure fuel; accelerate renewables and storage.",
        "Demand-side management; upgrade distribution to cut losses."
    ],
    "Factories in operation": [
        "Resolve power/logistics; single-window clearances; MSME credit guarantees.",
        "Skill programs; cluster common facilities; targeted tariff incentives."
    ],
    "Projects completed": [
        "Tighten monitoring with milestones; unblock land/clearances; link payments to progress.",
        "Reallocate to shovel-ready projects; enforce performance bonds."
    ],
    "New investment projects": [
        "Investor outreach; PPP pipeline; viability gap support for priority sectors.",
        "Stable policy and faster approvals; publish incentive calendars."
    ],
    "NSVA at basic prices": [
        "Counter-cyclical spending/capex; support services–industry linkages.",
        "Ease of doing business; export promotion; workforce upskilling."
    ],
    "Per capita NSDP": [
        "Stimulate labor-intensive sectors; targeted transfers for vulnerable.",
        "Human capital: training, apprenticeships, digital enablement."
    ],
    "_generic": [
        "Diagnose root causes vs 3-year baseline; quantify demand/supply/policy/shock factors.",
        "Deploy targeted interventions and track monthly KPIs for recovery."
    ]
}

RULES = {
    "Share of agriculture in NSVA": {"drop_pct": 5},
    "Share of industry in NSVA": {"drop_pct": 5},
    "Share of services in NSVA": {"drop_pct": 5},
    "NSVA at basic prices": {"drop_pct": 5},
    "Per capita NSDP": {"drop_pct": 5},
    "Gross cropped area": {"drop_pct": 5},
    "Cropping intensity": {"drop_pct": 5},
    "Air passenger traffic per million population": {"drop_pct": 8},
    "Road accidents": {"drop_pct": 8},
    "Electricity generation": {"drop_pct": 8},
    "Per capita energy consumption": {"drop_pct": 8},
    "Factories in operation": {"drop_pct": 5},
    "Gross value of output": {"drop_pct": 5},
    "Projects completed": {"drop_pct": 10},
    "New investment projects": {"drop_pct": 15},
    "_generic": {"drop_pct": 7.5}
}

# ------------------ Helpers ------------------

def parse_year(y):
    """Parse '2015-16' -> 2016 (end year), '2017' -> 2017."""
    if pd.isna(y):
        return np.nan
    s = str(y)
    if "-" in s:
        try:
            start = int(s[:4]); end2 = int(s[-2:])
            end = start // 100 * 100 + end2
            if end < start:
                end += 100
            return end
        except Exception:
            pass
    try:
        return int(s[:4])
    except Exception:
        return np.nan


def to_fy_period_index_from_endyears(idx, fiscal_end="MAR"):
    years = pd.Index([int(x) for x in idx])
    return pd.PeriodIndex(year=years, freq=f"A-{fiscal_end}")


def safe_num(x):
    try:
        xf = float(x)
    except Exception:
        return None
    return xf if np.isfinite(xf) else None


def clean_array(a, fill):
    arr = np.asarray(a, dtype=float)
    bad = ~np.isfinite(arr)
    if bad.any():
        arr[bad] = fill
    return arr

# ------------------ Forecast models ------------------

def naive_forecast(s, h=2):
    return np.array([s.iloc[-1]] * h)

def linear_forecast(s, h=2):
    x = np.arange(len(s)); y = s.values
    if len(s) < 2:
        return naive_forecast(s, h)
    b, a = np.polyfit(x, y, 1)
    xf = np.arange(len(s), len(s) + h)
    return a + b * xf

def sarimax_forecast(s, h=2):
    if len(s) < 6:
        return linear_forecast(s, h)
    s2 = s.copy()
    try:
        s2.index = to_fy_period_index_from_endyears(s2.index, fiscal_end="MAR")
    except Exception:
        s2.index = pd.period_range(start="2000", periods=len(s2), freq="A-MAR")
    m = SARIMAX(s2, order=(1, 1, 0), trend="t",
                enforce_stationarity=False, enforce_invertibility=False)
    res = m.fit(disp=False)
    return res.get_forecast(steps=h).predicted_mean.values

def ensemble_forecast(s, h=2):
    s = s.dropna()
    models = [("naive", naive_forecast), ("linear", linear_forecast)]
    if len(s) >= 6:
        models.append(("sarimax", sarimax_forecast))

    errs = {}
    start_bt = max(4, 6 if len(s) >= 6 else 4)
    for name, fn in models:
        e = []
        for t in range(start_bt, len(s)):
            train = s.iloc[:t]
            try:
                pred = fn(train, 1)[0]
            except Exception:
                pred = train.iloc[-1]
            y = s.iloc[t]
            denom = abs(y) if abs(y) > 1e-9 else 1.0
            e.append(abs(y - pred) / denom)
        errs[name] = np.mean(e) if e else 1.0

    inv = {k: 1 / (v + 1e-9) for k, v in errs.items()}
    sw = sum(inv.values()); weights = {k: v / sw for k, v in inv.items()}

    fc = np.zeros(h)
    for name, fn in models:
        try:
            part = fn(s, h)
        except Exception:
            part = naive_forecast(s, h)
        fc += weights[name] * part
    return fc, weights

def try_var(df_num, target, h=2):
    df_num = df_num.dropna()
    if df_num.shape[0] < 6 or df_num.shape[1] < 2 or target not in df_num.columns:
        return None
    try:
        model = VAR(df_num); res = model.fit(1)
        fc = res.forecast(df_num.values[-1:], steps=h)
        t_idx = list(df_num.columns).index(target)
        return fc[:, t_idx]
    except Exception:
        return None

# ------------------ Core analysis ------------------

def analyze_dataframe(df):
    """Return {metrics, correlations, series, columns} for the frontend."""
    if "FiscalEndYear" not in df.columns:
        df["FiscalEndYear"] = df["Year"].apply(parse_year) if "Year" in df.columns else pd.Series([np.nan] * len(df))
    df = df.sort_values("FiscalEndYear").set_index("FiscalEndYear")
    try:
        df.index = to_fy_period_index_from_endyears(df.index, fiscal_end="MAR")
    except Exception:
        pass

    cols = [c for c in df.columns if c not in ["Year", "FiscalEndYear"]]
    num = df[cols].apply(pd.to_numeric, errors="coerce").ffill(limit=1)

    metrics = []
    series = {}

    for c in cols:
        s = num[c].dropna()

        # last 10 points for chart
        if len(s):
            tail = s.iloc[-10:]
            ser = []
            for idx, val in tail.items():
                try:
                    y = int(getattr(idx, "year", int(idx)))
                except Exception:
                    y = None
                ser.append({"year": y, "value": safe_num(val)})
            series[c] = ser

        if len(s) < 3:
            metrics.append({"metric": c, "status": "insufficient"})
            continue

        fc, w = ensemble_forecast(s, h=2)
        var_fc = try_var(num[cols], c, 2)
        if var_fc is not None and np.isfinite(var_fc).all():
            fc = 0.7 * fc + 0.3 * var_fc
        fc = clean_array(fc, fill=s.iloc[-1])

        latest_year_raw = s.index[-1]
        try:
            latestYear = int(getattr(latest_year_raw, "year", int(latest_year_raw)))
        except Exception:
            latestYear = None

        latestVal = float(s.iloc[-1])
        base = float(s.iloc[-min(3, len(s)):].mean())
        dropBase = (latestVal - base) / base * 100 if base != 0 else 0.0
        dropFc = (latestVal - fc[0]) / fc[0] * 100 if fc[0] != 0 else 0.0

        rule = RULES.get(c, RULES["_generic"])
        th = -abs(rule["drop_pct"])
        decline = (dropBase <= th) or (dropFc <= th)

        sev = "Low"
        if decline:
            if (dropBase <= 2 * th) or (dropFc <= 2 * th):
                sev = "High"
            elif (dropBase <= 1.5 * th) or (dropFc <= 1.5 * th):
                sev = "Medium"

        metrics.append({
            "metric": c,
            "latestYear": latestYear,
            "latestValue": safe_num(latestVal),
            "forecast_t1": safe_num(fc[0]),
            "forecast_t2": safe_num(fc[1]),
            "drop_vs_baseline_pct": safe_num(dropBase),
            "drop_vs_forecast_pct": safe_num(dropFc),
            "declineFlag": bool(decline),
            "severity": sev,
            "actions": MASTER_ACTIONS.get(c, MASTER_ACTIONS["_generic"]),
            "weights": {k: safe_num(v) for k, v in (w or {}).items()}
        })

    # correlations
    corr_pairs = []
    valid = {c: num[c].dropna().values for c in cols if num[c].notna().sum() >= 5}
    keys = list(valid.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            n = min(len(valid[a]), len(valid[b]))
            if n < 5:
                continue
            xa = valid[a][-n:]; yb = valid[b][-n:]
            pr = np.corrcoef(xa, yb)[0, 1]
            ra = pd.Series(xa).rank().values
            rb = pd.Series(yb).rank().values
            sr = np.corrcoef(ra, rb)[0, 1]
            if not (np.isfinite(pr) and np.isfinite(sr)):
                continue
            corr_pairs.append({"x": a, "y": b, "pearson": safe_num(pr), "spearman": safe_num(sr), "n": int(n)})
    corr_pairs.sort(key=lambda r: abs(r["pearson"] or 0.0), reverse=True)
    corr_pairs = corr_pairs[:50]

    return {"metrics": metrics, "correlations": corr_pairs, "series": series, "columns": cols}

# ------------------ Endpoints ------------------

@app.get("/states")
def list_states():
    """
    Return all States/UTs with availability flag.
    You currently have data for Andhra_Pradesh and Andaman_Nicobar.
    """
    out = []
    for key, label in ALL_REGIONS.items():
        fname = STATE_FILES.get(key)
        available = False
        if fname:
            path = os.path.join(DATA_DIR, fname)
            available = os.path.exists(path)
        out.append({"key": key, "label": label, "available": bool(available)})
    return {"states": out}

@app.get("/analyze_state")
def analyze_state(state: str = Query(..., description="State/UT key from /states")):
    if state not in ALL_REGIONS:
        raise HTTPException(status_code=404, detail="Unknown state/UT key")
    fname = STATE_FILES.get(state)
    if not fname:
        raise HTTPException(status_code=404, detail="Data not available yet for this region")
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Data file not found on server")
    df = pd.read_excel(path, sheet_name=0)
    return analyze_dataframe(df)
