# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import math

app = FastAPI(
    title="Allo Towers API",
    description="Signal and FCC Tower Assessment Tool (distances in miles)"
)

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
    radius: float  # miles

class TowerAnalysis(BaseModel):
    opencellid_count: int
    fcc_count: int
    total_towers: int
    total_signal_samples: int
    opencellid_high_sample_count: int
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
OC_SIGNAL_COL = "averageSignal"
FCC_LAT_COL = "Lat"
FCC_LON_COL = "Lon"

# Earth's radius in miles
EARTH_RADIUS_MILES = 3958.8

# ---------- Utils ----------

def _vectorized_haversine(lat1_deg: float, lon1_deg: float,
                          lat2_deg: np.ndarray, lon2_deg: np.ndarray) -> np.ndarray:
    """Return distance in miles between (lat1, lon1) and arrays of points."""
    lat1 = np.radians(lat1_deg)
    lon1 = np.radians(lon1_deg)
    lat2 = np.radians(lat2_deg)
    lon2 = np.radians(lon2_deg)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_MILES * c

def _bounding_box(lat0: float, lon0: float, radius_miles: float) -> tuple[float, float, float, float]:
    """
    Approximate bounding box in degrees for a given radius in miles.
    1° latitude ≈ 69 miles; longitude scaled by cos(lat).
    """
    lat_delta = radius_miles / 69.0
    lon_scale = max(math.cos(math.radians(lat0)), 1e-6)
    lon_delta = radius_miles / (69.0 * lon_scale)
    return (lat0 - lat_delta, lat0 + lat_delta, lon0 - lon_delta, lon0 + lon_delta)

def _prep_opencellid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[OC_LAT_COL] = pd.to_numeric(df[OC_LAT_COL], errors="coerce")
    df[OC_LON_COL] = pd.to_numeric(df[OC_LON_COL], errors="coerce")
    if OC_SAMPLES_COL in df.columns:
        df[OC_SAMPLES_COL] = (
            pd.to_numeric(df[OC_SAMPLES_COL], errors="coerce")
              .fillna(0)
              .astype(int)
        )
    else:
        df[OC_SAMPLES_COL] = 0
    df = df.dropna(subset=[OC_LAT_COL, OC_LON_COL]).reset_index(drop=True)
    return df

def _prep_fcc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[FCC_LAT_COL] = pd.to_numeric(df[FCC_LAT_COL], errors="coerce")
    df[FCC_LON_COL] = pd.to_numeric(df[FCC_LON_COL], errors="coerce")
    df = df.dropna(subset=[FCC_LAT_COL, FCC_LON_COL]).reset_index(drop=True)
    return df

def load_data():
    global opencellid_df, fcc_df
    here = Path(__file__).resolve().parent
    data_dir = (here / ".." / ".." / "data").resolve()

    oc = pd.read_csv(data_dir / "Signal Dataset.csv")
    fcc = pd.read_csv(data_dir / "FCC_towers.csv")
    oc.columns = oc.columns.str.strip()
    fcc.columns = fcc.columns.str.strip()

    opencellid_df = _prep_opencellid(oc)
    fcc_df = _prep_fcc(fcc)

@app.on_event("startup")
async def startup_event():
    load_data()

# ---------- Endpoints ----------

@app.get("/")
async def root():
    return {"message": "Allo Towers API - distances in miles"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "data_loaded": (opencellid_df is not None and fcc_df is not None),
        "opencellid_records": len(opencellid_df) if opencellid_df is not None else 0,
        "fcc_records": len(fcc_df) if fcc_df is not None else 0,
    }

@app.post("/analyze_towers", response_model=TowerAnalysis)
async def analyze_towers(req: LocationRequest):
    if opencellid_df is None or fcc_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    lat0, lon0, radius = req.latitude, req.longitude, req.radius

    # OpenCellID subset
    lat_min, lat_max, lon_min, lon_max = _bounding_box(lat0, lon0, radius)
    oc_subset = opencellid_df[
        opencellid_df[OC_LAT_COL].between(lat_min, lat_max) &
        opencellid_df[OC_LON_COL].between(lon_min, lon_max)
    ]
    if not oc_subset.empty:
        oc_dist = _vectorized_haversine(lat0, lon0,
                                        oc_subset[OC_LAT_COL].values,
                                        oc_subset[OC_LON_COL].values)
        mask = oc_dist <= radius
        oc_subset = oc_subset.loc[mask].copy()
        oc_subset["distance_miles"] = np.round(oc_dist[mask], 2)
    else:
        oc_subset = oc_subset.assign(distance_miles=pd.Series(dtype=float))

    total_signal_samples = int(oc_subset[OC_SAMPLES_COL].sum())
    oc_high_count = int((oc_subset[OC_SAMPLES_COL] > 100).sum())

    if OC_SIGNAL_COL in oc_subset.columns:
        sig_vals = pd.to_numeric(oc_subset[OC_SIGNAL_COL], errors="coerce").dropna()
        if not sig_vals.empty:
            max_signal = float(sig_vals.max())
            min_signal = float(sig_vals.min())
            avg_signal = float(sig_vals.mean())
            std_signal = float(sig_vals.std())
        else:
            max_signal = min_signal = avg_signal = std_signal = 0.0
    else:
        max_signal = min_signal = avg_signal = std_signal = 0.0

    oc_records = []
    for r in oc_subset.itertuples(index=False):
        row = r._asdict() if hasattr(r, "_asdict") else dict(r._mapping)
        oc_records.append({
            "radio": row.get("radio", ""),
            "mcc": row.get("mcc", ""),
            "net": row.get("net", ""),
            "area": row.get("area", ""),
            "cell": row.get("cell", ""),
            "lat": row[OC_LAT_COL],
            "lon": row[OC_LON_COL],
            "range": row.get("range", ""),
            "samples": int(row.get(OC_SAMPLES_COL, 0)),
            "averageSignal": row.get("averageSignal", ""),
            "distance_miles": row["distance_miles"],
        })

    # FCC subset
    lat_min, lat_max, lon_min, lon_max = _bounding_box(lat0, lon0, radius)
    fcc_subset = fcc_df[
        fcc_df[FCC_LAT_COL].between(lat_min, lat_max) &
        fcc_df[FCC_LON_COL].between(lon_min, lon_max)
    ]
    if not fcc_subset.empty:
        fcc_dist = _vectorized_haversine(lat0, lon0,
                                         fcc_subset[FCC_LAT_COL].values,
                                         fcc_subset[FCC_LON_COL].values)
        mask = fcc_dist <= radius
        fcc_subset = fcc_subset.loc[mask].copy()
        fcc_subset["distance_miles"] = np.round(fcc_dist[mask], 2)
    else:
        fcc_subset = fcc_subset.assign(distance_miles=pd.Series(dtype=float))

    fcc_records = []
    for r in fcc_subset.itertuples(index=False):
        row = r._asdict() if hasattr(r, "_asdict") else dict(r._mapping)
        fcc_records.append({
            "file_number": row.get("File Number_x", ""),
            "registration_number": row.get("Registration Number", ""),
            "structure_type": row.get("Structure Type", ""),
            "height": row.get("Height of Structure", ""),
            "ground_elevation": row.get("Ground Elevation", ""),
            "overall_height": row.get("Overall Height Above Ground", ""),
            "lat": row[FCC_LAT_COL],
            "lon": row[FCC_LON_COL],
            "city": row.get("Structure_City", ""),
            "state": row.get("Structure_State Code", ""),
            "distance_miles": row["distance_miles"],
        })

    return TowerAnalysis(
        opencellid_count=len(oc_records),
        fcc_count=len(fcc_records),
        total_towers=len(oc_records) + len(fcc_records),
        total_signal_samples=total_signal_samples,
        opencellid_high_sample_count=oc_high_count,
        max_signal_strength=max_signal,
        min_signal_strength=min_signal,
        avg_signal_strength=avg_signal,
        std_signal_strength=std_signal,
        opencellid_towers=oc_records,
        fcc_towers=fcc_records,
    )

@app.get("/data_info")
async def get_data_info():
    if opencellid_df is None or fcc_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    return {
        "opencellid_records": len(opencellid_df),
        "fcc_records": len(fcc_df),
        "opencellid_columns": list(opencellid_df.columns),
        "fcc_columns": list(fcc_df.columns),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
