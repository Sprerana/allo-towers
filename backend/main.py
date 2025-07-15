from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import math
import os

app = FastAPI(title="Allo Towers API", description="Signal and FCC Tower Assessment Tool")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class LocationRequest(BaseModel):
    latitude: float
    longitude: float
    radius: float  # in kilometers

class TowerAnalysis(BaseModel):
    opencellid_count: int
    fcc_count: int
    total_towers: int
    total_signal_samples: int
    opencellid_high_sample_count: int
    opencellid_towers: List[Dict[str, Any]]
    fcc_towers: List[Dict[str, Any]]

# Global variables to store loaded data
opencellid_data = None
fcc_data = None

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    
    return c * r

def load_data():
    """Load the tower datasets"""
    global opencellid_data, fcc_data
    
    try:
        # Load OpenCellID data
        opencellid_data = pd.read_csv('../data/Signal Dataset.csv')
        print(f"Loaded OpenCellID data: {len(opencellid_data)} records")
        
        # Load FCC data
        fcc_data = pd.read_csv('../data/FCC_towers.csv')
        print(f"Loaded FCC data: {len(fcc_data)} records")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load data when the application starts"""
    load_data()

@app.get("/")
async def root():
    return {"message": "Allo Towers API - Signal and FCC Tower Assessment Tool"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "data_loaded": opencellid_data is not None and fcc_data is not None}

@app.post("/analyze_towers", response_model=TowerAnalysis)
async def analyze_towers(request: LocationRequest):
    """
    Analyze towers within the specified radius of the given location
    """
    if opencellid_data is None or fcc_data is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    try:
        # Filter OpenCellID towers within radius
        opencellid_distances = []
        opencellid_towers_in_radius = []
        total_signal_samples = 0
        opencellid_high_sample_count = 0
        
        for _, tower in opencellid_data.iterrows():
            try:
                lat = float(tower['lat'])
                lon = float(tower['lon'])
                distance = haversine_distance(request.latitude, request.longitude, lat, lon)
                
                if distance <= request.radius:
                    # Get samples count
                    samples = tower.get('samples',0)
                    print("sample number is",samples)
                    try:
                        samples_int = int(samples) if samples != '' else 0
                        total_signal_samples += samples_int
                        if samples_int > 100:
                            opencellid_high_sample_count += 1
                    except (ValueError, TypeError):
                        samples_int = 0
                    
                    opencellid_towers_in_radius.append({
                        'radio': tower.get('radio', ''),
                        'mcc': tower.get('mcc', ''),
                        'net': tower.get('net', ''),
                        'area': tower.get('area', ''),
                        'cell': tower.get('cell', ''),
                        'lat': lat,
                        'lon': lon,
                        'range': tower.get('range', ''),
                        'samples': samples_int,
                        'averageSignal': tower.get('averageSignal', ''),
                        'distance_km': round(distance, 2)
                    })
            except (ValueError, KeyError):
                continue
        
        # Filter FCC towers within radius
        fcc_towers_in_radius = []
        
        for _, tower in fcc_data.iterrows():
            try:
                lat = float(tower['Lat'])
                lon = float(tower['Lon'])
                distance = haversine_distance(request.latitude, request.longitude, lat, lon)
                
                if distance <= request.radius:
                    fcc_towers_in_radius.append({
                        'file_number': tower.get('File Number_x', ''),
                        'registration_number': tower.get('Registration Number', ''),
                        'structure_type': tower.get('Structure Type', ''),
                        'height': tower.get('Height of Structure', ''),
                        'ground_elevation': tower.get('Ground Elevation', ''),
                        'overall_height': tower.get('Overall Height Above Ground', ''),
                        'lat': lat,
                        'lon': lon,
                        'city': tower.get('Structure_City', ''),
                        'state': tower.get('Structure_State Code', ''),
                        'distance_km': round(distance, 2)
                    })
            except (ValueError, KeyError):
                continue
        
        return TowerAnalysis(
            opencellid_count=len(opencellid_towers_in_radius),
            fcc_count=len(fcc_towers_in_radius),
            total_towers=len(opencellid_towers_in_radius) + len(fcc_towers_in_radius),
            total_signal_samples=total_signal_samples,
            opencellid_high_sample_count=opencellid_high_sample_count,
            opencellid_towers=opencellid_towers_in_radius,
            fcc_towers=fcc_towers_in_radius
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing towers: {str(e)}")

@app.get("/data_info")
async def get_data_info():
    """Get information about the loaded datasets"""
    if opencellid_data is None or fcc_data is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    return {
        "opencellid_records": len(opencellid_data),
        "fcc_records": len(fcc_data),
        "opencellid_columns": list(opencellid_data.columns),
        "fcc_columns": list(fcc_data.columns),
        #"opencellid_sample": opencellid_data.head(3).to_dict('records') if len(opencellid_data) > 0 else [],
        #"fcc_sample": fcc_data.head(3).to_dict('records') if len(fcc_data) > 0 else []
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)