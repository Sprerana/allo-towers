
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any
import json

# Page configuration
st.set_page_config(
    page_title="Allo Towers - Signal and FCC Tower Assessment Tool",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .tower-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

##API configuration
API_BASE_URL = "http://localhost:8080"

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_data_info():
    """Get information about the loaded datasets"""
    try:
        response = requests.get(f"{API_BASE_URL}/data_info", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def analyze_towers(latitude: float, longitude: float, radius: float):
    """Send analysis request to the API"""
    try:
        payload = {
            "latitude": latitude,
            "longitude": longitude,
            "radius": radius
        }
        response = requests.post(f"{API_BASE_URL}/analyze_towers", json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def create_tower_map(towers_data: Dict[str, Any], center_lat: float, center_lon: float):
    """Create a map visualization of the towers"""
    if not towers_data:
        return None
    
    # Prepare data for plotting
    opencellid_towers = towers_data.get('opencellid_towers', [])
    fcc_towers = towers_data.get('fcc_towers', [])
    
    # Create figure
    fig = go.Figure()
    
    # Add OpenCellID towers
    if opencellid_towers:
        opencellid_df = pd.DataFrame(opencellid_towers)
        fig.add_trace(go.Scattermapbox(
            lat=opencellid_df['lat'],
            lon=opencellid_df['lon'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=8,
                color='blue',
                opacity=0.7
            ),
            text=opencellid_df.apply(lambda row: f"OpenCellID<br>Distance: {row['distance_km']} km<br>Radio: {row['radio']}", axis=1),
            hoverinfo='text',
            name='OpenCellID Towers'
        ))
    
    # Add FCC towers
    if fcc_towers:
        fcc_df = pd.DataFrame(fcc_towers)
        fig.add_trace(go.Scattermapbox(
            lat=fcc_df['lat'],
            lon=fcc_df['lon'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=8,
                color='red',
                opacity=0.7
            ),
            text=fcc_df.apply(lambda row: f"FCC Tower<br>Distance: {row['distance_km']} km<br>Type: {row['structure_type']}", axis=1),
            hoverinfo='text',
            name='FCC Towers'
        ))
    
    # Add center point
    fig.add_trace(go.Scattermapbox(
        lat=[center_lat],
        lon=[center_lon],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=15,
            color='green',
            symbol='star'
        ),
        text=['Search Center'],
        hoverinfo='text',
        name='Search Center'
    ))
    
    # Update layout
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=10
        ),
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">Allo Towers</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #666;">Signal and FCC Tower Assessment Tool</h2>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ Backend API is not running. Please start the FastAPI server first.")
        st.info("To start the backend, run: `cd backend && python main.py`")
        return
    
    # Get data info
    data_info = get_data_info()
    if data_info:
        st.sidebar.success("✅ Backend API is running")
        st.sidebar.info(f"Total OpenCellID Records: {data_info['opencellid_records']:,}")
        st.sidebar.info(f"🏗️ Total FCC Records: {data_info['fcc_records']:,}")
    else:
        st.sidebar.warning("⚠️ Could not retrieve data information")
    
    # Input form
    st.markdown("### 📍 Location and Search Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=40.7128,
            step=0.0001,
            format="%.4f",
            help="Enter latitude in decimal degrees (e.g., 40.7128 for New York)"
        )
    
    with col2:
        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=-74.0060,
            step=0.0001,
            format="%.4f",
            help="Enter longitude in decimal degrees (e.g., -74.0060 for New York)"
        )
    
    with col3:
        radius = st.number_input(
            "Search Radius (km)",
            min_value=0.1,
            max_value=100.0,
            value=10.0,
            step=0.1,
            format="%.1f",
            help="Enter search radius in kilometers"
        )
    
    # Analysis button
    if st.button("🔍 Analyze Towers", type="primary", use_container_width=True):
        with st.spinner("Analyzing towers in the specified area..."):
            results = analyze_towers(latitude, longitude, radius)
            
            if results:
                #st.success("✅ Analysis completed!")
                
                # Display metrics
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
                
                
                # Additional metrics
                st.markdown("### 📊 Signal Analysis Metrics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Signal Samples", f"{results['total_signal_samples']:,}")
                
                with col2:
                    st.metric("High Sample Towers (>100)", results['opencellid_high_sample_count'])
                
                with col3:
                    avg_samples = results['total_signal_samples'] / results['opencellid_count'] if results['opencellid_count'] > 0 else 0
                    st.metric("Avg Samples per Tower", f"{avg_samples:.1f}")
                

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
                
                # Create and display map
                st.markdown("### 🗺️ Tower Locations")
                map_fig = create_tower_map(results, latitude, longitude)
                if map_fig:
                    st.plotly_chart(map_fig, use_container_width=True)
                
                # Signal Analysis Visualizations
                if results['opencellid_towers']:
                    opencellid_df = pd.DataFrame(results['opencellid_towers'])
                    
                    # Radio type analysis
                    if 'radio' in opencellid_df.columns:
                        st.markdown("#### 📱 Radio Type Analysis")
                        radio_stats = opencellid_df.groupby('radio').agg({
                            'samples': ['count', 'sum', 'mean'],
                            'distance_km': 'mean'
                        }).round(2)
                        radio_stats.columns = ['Tower Count', 'Total Samples', 'Avg Samples', 'Avg Distance (km)']
                        st.dataframe(radio_stats, use_container_width=True)
                    
                    st.markdown("#### 📶 Network Type Distribution")  # added
                    network_df = opencellid_df['radio'].value_counts().reset_index()  # added
                    network_df.columns = ['Network Type', 'Count']  # added
                    fig_network = px.bar(  # added
                            network_df,
                            x='Network Type',
                            y='Count',
                            title='Network Type Distribution Within Radius',  # added
                            labels={'Network Type': 'Network Type', 'Count': 'Number of Towers'}  # added
                        )  # added
                    st.plotly_chart(fig_network, use_container_width=True) 
                
                # Display detailed results
                if results['opencellid_towers'] or results['fcc_towers']:
                    st.markdown("### 📋 Detailed Results")
                    
                    # OpenCellID Towers
                    if results['opencellid_towers']:
                        st.markdown("#### 📱 OpenCellID Towers")
                        opencellid_df = pd.DataFrame(results['opencellid_towers'])
                        st.dataframe(
                            opencellid_df[['radio', 'mcc', 'net', 'lat', 'lon', 'range', 'samples', 'distance_km']],
                            use_container_width=True
                        )
                    
                    # FCC Towers
                    if results['fcc_towers']:
                        st.markdown("#### 🏗️ FCC Towers")
                        fcc_df = pd.DataFrame(results['fcc_towers'])
                        st.dataframe(
                            fcc_df[['structure_type', 'height', 'lat', 'lon', 'city', 'state', 'distance_km']],
                            use_container_width=True
                        )
                else:
                    st.info("No towers found within the specified radius.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; font-size: 0.8rem;">
            Allo Towers - Signal and FCC Tower Assessment Tool<br>
            Built with Streamlit and FastAPI
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()