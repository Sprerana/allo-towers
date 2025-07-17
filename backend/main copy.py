### main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import math

app = FastAPI(title="Allo Towers API", description="Signal and FCC Tower Assessment Tool")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class LocationRequest(BaseModel):
    latitude: float
    longitude: float
    radius: float  # km

class TowerAnalysis(BaseModel):
    opencellid_count: int
    fcc_count: int
    total_towers: int
    total_signal_samples: int
    opencellid_high_sample_count: int
    # Added: signal strength metrics
    max_signal_strength: float
    min_signal_strength: float
    avg_signal_strength: float
    std_signal_strength: float
    opencellid_towers: List[Dict[str, Any]]
    fcc_towers: List[Dict[str, Any]]

# ---------- Globals ----------
opencellid_df: pd.DataFrame | None = None
fcc_df: pd.DataFrame | None = None

# Columns
OC_LAT_COL = "lat"
OC_LON_COL = "lon"
OC_SAMPLES_COL = "samples"
# The Signal Dataset.csv has an averageSignal column
OC_SIGNAL_COL = "averageSignal"  # Added: reference to averageSignal
FCC_LAT_COL = "Lat"
FCC_LON_COL = "Lon"
EARTH_RADIUS_KM = 6371.0

# ---------- Utils ----------
# ... (unchanged utility functions) ...

def load_data():
    global opencellid_df, fcc_df
    here = Path(__file__).resolve().parent
    data_dir = (here / ".." / ".." / "data").resolve()

    oc_path = data_dir / "Signal Dataset.csv"
    fcc_path = data_dir / "FCC_towers.csv"

    oc = pd.read_csv(oc_path)
    fcc = pd.read_csv(fcc_path)

    opencellid_df = _prep_opencellid(oc)
    fcc_df = _prep_fcc(fcc)

    print(f"Loaded OpenCellID data: {len(opencellid_df)} records (cleaned).")
    print(f"Loaded FCC data: {len(fcc_df)} records (cleaned).")

@app.on_event("startup")
async def startup_event():
    load_data()

# ---------- Endpoints ----------

@app.post("/analyze_towers", response_model=TowerAnalysis)
async def analyze_towers(req: LocationRequest):
    if opencellid_df is None or fcc_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    lat0, lon0, radius = req.latitude, req.longitude, req.radius

    # OpenCellID filtering (unchanged)
    lat_min, lat_max, lon_min, lon_max = _bounding_box(lat0, lon0, radius)
    oc_subset = opencellid_df[
        opencellid_df[OC_LAT_COL].between(lat_min, lat_max) &
        opencellid_df[OC_LON_COL].between(lon_min, lon_max)
    ]
    if not oc_subset.empty:
        oc_dist = _vectorized_haversine(lat0, lon0, oc_subset[OC_LAT_COL].values, oc_subset[OC_LON_COL].values)
        mask = oc_dist <= radius
        oc_subset = oc_subset.loc[mask].copy()
        oc_subset["distance_km"] = np.round(oc_dist[mask], 2)
    else:
        oc_subset = oc_subset.assign(distance_km=pd.Series(dtype=float))

    # Compute signal sample metrics (unchanged)
    total_signal_samples = int(oc_subset[OC_SAMPLES_COL].sum()) if OC_SAMPLES_COL in oc_subset else 0
    oc_high_count = int((oc_subset[OC_SAMPLES_COL] > 100).sum()) if OC_SAMPLES_COL in oc_subset else 0

    # Added: compute signal strength metrics
    if OC_SIGNAL_COL in oc_subset.columns:
        oc_subset[OC_SIGNAL_COL] = pd.to_numeric(oc_subset[OC_SIGNAL_COL], errors='coerce')
        sig_vals = oc_subset[OC_SIGNAL_COL].dropna()
        if not sig_vals.empty:
            max_signal = float(sig_vals.max())
            min_signal = float(sig_vals.min())
            avg_signal = float(sig_vals.mean())
            std_signal = float(sig_vals.std())
        else:
            max_signal = min_signal = avg_signal = std_signal = 0.0
    else:
        max_signal = min_signal = avg_signal = std_signal = 0.0

    # Prepare record lists (unchanged)...
    oc_records = []
    for r in oc_subset.itertuples(index=False):
        row = r._asdict() if hasattr(r, '_asdict') else dict(r._mapping)
        oc_records.append({
            'radio': row.get('radio', ''),
            'mcc': row.get('mcc', ''),
            'net': row.get('net', ''),
            'area': row.get('area', ''),
            'cell': row.get('cell', ''),
            'lat': row[OC_LAT_COL],
            'lon': row[OC_LON_COL],
            'range': row.get('range', ''),
            'samples': int(row.get(OC_SAMPLES_COL, 0)),
            'averageSignal': row.get(OC_SIGNAL_COL, ''),  # Ensures signal is returned
            'distance_km': row['distance_km'],
        })

    # FCC processing (unchanged)...
    # ...

    # Return including new metrics
    return TowerAnalysis(
        opencellid_count=len(oc_records),
        fcc_count=len(fcc_records),
        total_towers=len(oc_records) + len(fcc_records),
        total_signal_samples=total_signal_samples,
        opencellid_high_sample_count=oc_high_count,
        max_signal_strength=max_signal,  # Added
        min_signal_strength=min_signal,  # Added
        avg_signal_strength=avg_signal,  # Added
        std_signal_strength=std_signal,  # Added
        opencellid_towers=oc_records,
        fcc_towers=fcc_records,
    )

# Other endpoints unchanged...

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


### app.py

import streamlit as st
import requests
import pandas as pd
import numpy as np  # Added for any numeric operations
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any
import json

# Page configuration (unchanged)
# ...

# API configuration
API_BASE_URL = "http://localhost:8000"

# Helper functions (unchanged)
# ...

def main():
    # Header and health check (unchanged)
    # ...
    if st.button("🔍 Analyze Towers", type="primary", use_container_width=True):
        with st.spinner("Analyzing towers in the specified area..."):
            results = analyze_towers(latitude, longitude, radius)
            if results:
                st.success("✅ Analysis completed!")

                # Display existing metrics (unchanged)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Towers", results['total_towers'])
                with col2:
                    st.metric("OpenCellID Towers", results['opencellid_count'])
                with col3:
                    st.metric("FCC Towers", results['fcc_count'])
                with col4:
                    density = results['total_towers'] / (3.14159 * radius * radius) if radius > 0 else 0
                    st.metric("Tower Density", f"{density:.2f} towers/km²")

                # Signal sample metrics (unchanged)
                st.markdown("### 📊 Signal Analysis Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Signal Samples", f"{results['total_signal_samples']:,}")
                with col2:
                    st.metric("High Sample Towers (>100)", results['opencellid_high_sample_count'])
                with col3:
                    avg_samples = results['total_signal_samples'] / results['opencellid_count'] if results['opencellid_count'] > 0 else 0
                    st.metric("Avg Samples per Tower", f"{avg_samples:.1f}")
                with col4:
                    high_sample_percentage = (results['opencellid_high_sample_count'] / results['opencellid_count'] * 100) if results['opencellid_count'] > 0 else 0
                    st.metric("High Sample %", f"{high_sample_percentage:.1f}%")

                # Added: Signal strength metrics
                st.markdown("### 🚦 Signal Strength Metrics")
                col5, col6, col7, col8 = st.columns(4)
                with col5:
                    st.metric("Max Signal Strength", f"{results.get('max_signal_strength', 0.0):.2f}")
                with col6:
                    st.metric("Min Signal Strength", f"{results.get('min_signal_strength', 0.0):.2f}")
                with col7:
                    st.metric("Avg Signal Strength", f"{results.get('avg_signal_strength', 0.0):.2f}")
                with col8:
                    st.metric("Std Dev Signal Strength", f"{results.get('std_signal_strength', 0.0):.2f}")

                # Continue with map and other visualizations (unchanged)
                # ...

if __name__ == "__main__":
    main()
