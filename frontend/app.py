# frontend/app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

# Page configuration
st.set_page_config(
    page_title="Allo Towers - Signal and FCC Tower Assessment Tool",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
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

API_BASE_URL = "http://localhost:8000"

def check_api_health() -> bool:
    try:
        return requests.get(f"{API_BASE_URL}/health", timeout=5).status_code == 200
    except:
        return False

def get_data_info() -> dict | None:
    try:
        r = requests.get(f"{API_BASE_URL}/data_info", timeout=10)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

def analyze_towers(latitude: float, longitude: float, radius: float) -> dict | None:
    try:
        payload = {"latitude": latitude, "longitude": longitude, "radius": radius}
        r = requests.post(f"{API_BASE_URL}/analyze_towers", json=payload, timeout=30)
        if r.status_code == 200:
            return r.json()
        st.error(f"API Error: {r.text}")
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
    return None

def create_tower_map(towers_data: Dict[str, Any], center_lat: float, center_lon: float):
    fig = go.Figure()

    # OpenCellID Towers
    oc = towers_data.get("opencellid_towers", [])
    if oc:
        df_oc = pd.DataFrame(oc)
        fig.add_trace(go.Scattermapbox(
            lat=df_oc["lat"],
            lon=df_oc["lon"],
            mode="markers",
            marker=go.scattermapbox.Marker(size=8, color="blue", opacity=0.7),
            text=df_oc.apply(
                lambda r: f"OpenCellID<br>Distance: {r['distance_miles']} mi<br>Radio: {r['radio']}",
                axis=1
            ),
            hoverinfo="text",
            name="OpenCellID Towers"
        ))

    # FCC Towers
    fc = towers_data.get("fcc_towers", [])
    if fc:
        df_fc = pd.DataFrame(fc)
        fig.add_trace(go.Scattermapbox(
            lat=df_fc["lat"],
            lon=df_fc["lon"],
            mode="markers",
            marker=go.scattermapbox.Marker(size=8, color="red", opacity=0.7),
            text=df_fc.apply(
                lambda r: f"FCC Tower<br>Distance: {r['distance_miles']} mi<br>Type: {r['structure_type']}",
                axis=1
            ),
            hoverinfo="text",
            name="FCC Towers"
        ))

    # Search center
    fig.add_trace(go.Scattermapbox(
        lat=[center_lat],
        lon=[center_lon],
        mode="markers",
        marker=go.scattermapbox.Marker(size=15, color="green", symbol="star"),
        text=["Search Center"],
        hoverinfo="text",
        name="Search Center"
    ))

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
    st.markdown(
        '<h2 style="text-align: center; color: #666;">Signal and FCC Tower Assessment Tool</h2>',
        unsafe_allow_html=True
    )

    # Backend health check
    if not check_api_health():
        st.error("⚠️ Backend API is not running. Please start it with: `cd backend && python main.py`")
        return

    # Sidebar info
    info = get_data_info()
    if info:
        st.sidebar.success("✅ Backend API is running")
        st.sidebar.info(f"Total OpenCellID Records: {info['opencellid_records']:,}")
        st.sidebar.info(f"Total FCC Records: {info['fcc_records']:,}")
    else:
        st.sidebar.warning("⚠️ Could not retrieve data info")

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
            help="Enter latitude in decimal degrees (e.g., 40.7128)"
        )
    with col2:
        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=-74.0060,
            step=0.0001,
            format="%.4f",
            help="Enter longitude in decimal degrees (e.g., -74.0060)"
        )
    with col3:
        radius = st.number_input(
            "Search Radius (miles)",
            min_value=0.1,
            max_value=100.0,
            value=10.0,
            step=0.1,
            format="%.1f",
            help="Enter search radius in miles"
        )

    # Trigger analysis
    if st.button("🔍 Analyze Towers", type="primary", use_container_width=True):
        with st.spinner("Analyzing towers in the specified area..."):
            results = analyze_towers(latitude, longitude, radius)
            if results:
                # Top-line metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Towers", results["total_towers"])
                m2.metric("OpenCellID Towers", results["opencellid_count"])
                m3.metric("FCC Towers", results["fcc_count"])
                density = (
                    results["total_towers"] /
                    (3.14159 * radius * radius) if radius > 0 else 0
                )
                m4.metric("Tower Density", f"{density:.2f} towers/mi²")

                # Signal sample metrics
                st.markdown("### 📊 Signal Analysis Metrics")
                s1, s2, s3 = st.columns(3)
                s1.metric("Total Signal Samples", f"{results['total_signal_samples']:,}")
                s2.metric("High Sample Towers (>100)", results["opencellid_high_sample_count"])
                avg_samples = (
                    results["total_signal_samples"] / results["opencellid_count"]
                    if results["opencellid_count"] > 0 else 0
                )
                s3.metric("Avg Samples per Tower", f"{avg_samples:.1f}")

                # Map
                st.markdown("### 🗺️ Tower Locations")
                st.plotly_chart(
                    create_tower_map(results, latitude, longitude),
                    use_container_width=True
                )

                # Further per-radio analyses
                if results["opencellid_towers"]:
                    df_oc = pd.DataFrame(results["opencellid_towers"])

                    # Signal by network type
                    if "radio" in df_oc.columns and "averageSignal" in df_oc.columns:
                        st.markdown("#### 🚦 Signal Strength by Network Type")
                        df_oc["averageSignal"] = pd.to_numeric(
                            df_oc["averageSignal"], errors="coerce"
                        )
                        signal_stats = (
                            df_oc.groupby("radio")["averageSignal"]
                            .agg([
                                ("Max Signal (dBm)", "max"),
                                ("Min Signal (dBm)", "min"),
                                ("Avg Signal (dBm)", "mean"),
                                ("Std Dev Signal (dBm)", "std")
                            ])
                            .round(2)
                            .reset_index()
                        )
                        st.dataframe(signal_stats, use_container_width=True)

                    # Radio type summary
                    if "radio" in df_oc.columns:
                        st.markdown("#### 📱 Radio Type Analysis")
                        radio_stats = (
                            df_oc.groupby("radio")
                            .agg({
                                "samples": ["count", "sum", "mean"],
                                "distance_miles": "mean"
                            })
                            .round(2)
                        )
                        radio_stats.columns = [
                            "Tower Count",
                            "Total Samples",
                            "Avg Samples",
                            "Avg Distance (mi)"
                        ]
                        st.dataframe(radio_stats, use_container_width=True)

                    # Radio distribution
                    st.markdown("#### 📶 Network Type Distribution")
                    network_df = (
                        df_oc["radio"]
                        .value_counts()
                        .rename_axis("Network Type")
                        .reset_index(name="Count")
                    )
                    fig_network = px.bar(
                        network_df,
                        x="Network Type",
                        y="Count",
                        title="Network Type Distribution Within Radius",
                        labels={"Count": "Number of Towers"}
                    )
                    st.plotly_chart(fig_network, use_container_width=True)

                # Detailed tables
                st.markdown("### 📋 Detailed Results")
                if results["opencellid_towers"]:
                    st.markdown("#### 📱 OpenCellID Towers")
                    st.dataframe(
                        pd.DataFrame(results["opencellid_towers"])[
                            ["radio", "mcc", "net", "lat", "lon", "range", "samples", "distance_miles"]
                        ],
                        use_container_width=True
                    )
                if results["fcc_towers"]:
                    st.markdown("#### 🏗️ FCC Towers")
                    st.dataframe(
                        pd.DataFrame(results["fcc_towers"])[
                            ["structure_type", "height", "lat", "lon", "city", "state", "distance_miles"]
                        ],
                        use_container_width=True
                    )
                if not results["opencellid_towers"] and not results["fcc_towers"]:
                    st.info("No towers found within the specified radius.")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; font-size: 0.8rem;">
            Allo Towers — Built with Streamlit & FastAPI (distances in miles)
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
