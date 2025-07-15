# 📡 Allo Towers - Signal and FCC Tower Assessment Tool

A web-based assessment platform that enables users to analyze cellular signal quality and infrastructure density within a geographic radius using OpenCellID and FCC tower datasets.

## 🚀 Features

- **Geographic Analysis**: Input latitude, longitude, and search radius to analyze tower density
- **Dual Dataset Support**: Analyze both OpenCellID cellular towers and FCC registered towers
- **Signal Sample Analysis**: Count total signal samples and identify high-sample towers (>100 samples)
- **Interactive Map**: Visualize tower locations with an interactive map
- **Real-time Metrics**: Get instant counts, density calculations, and signal statistics
- **Data Visualizations**: Histograms, scatter plots, and radio type analysis
- **Modern UI**: Beautiful, responsive interface built with Streamlit

## 🏗️ Architecture

- **Frontend**: Streamlit web application
- **Backend**: FastAPI REST API
- **Data Processing**: Pandas for efficient data manipulation
- **Visualization**: Plotly for interactive maps and charts

## 📋 Prerequisites

- Python 3.11
- Conda (for environment management)
- The following datasets in the `data/` folder:
  - `Signal Dataset.csv` (OpenCellID data)
  - `FCC_towers.csv` (FCC tower data)

## 🛠️ Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd /Users/preranamandalika/allo-towers
   ```

2. **Create and activate the conda environment:**
   ```bash
   conda create -n allo-towers python=3.11 -y
   conda activate allo-towers
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure data files are in place:**
   ```bash
   ls data/
   # Should show: Signal Dataset.csv, FCC_towers.csv
   ```

## 🚀 Running the Application

### 1. Start the Backend API

In one terminal:
```bash
cd backend
python main.py
```

The FastAPI server will start on `http://localhost:8000`

### 2. Start the Frontend

In another terminal:
```bash
cd frontend
streamlit run app.py
```

The Streamlit app will open in your browser at `http://localhost:8501`

## 📊 Data Sources

### OpenCellID Dataset
- **Source**: OpenCellID project
- **Content**: Cellular tower information including radio type, MCC, MNC, coordinates, signal range, and samples
- **Columns**: radio, mcc, net, area, cell, unit, lon, lat, range, samples, changeable, created, updated, averageSignal

### FCC Towers Dataset
- **Source**: Federal Communications Commission (FCC)
- **Content**: Registered tower structures with detailed information
- **Key Columns**: File Number, Registration Number, Structure Type, Height, Coordinates, Location details

## 🎯 Usage

1. **Open the Streamlit application** in your browser
2. **Enter location parameters**:
   - Latitude (decimal degrees, e.g., 40.7128 for New York)
   - Longitude (decimal degrees, e.g., -74.0060 for New York)
   - Search radius in kilometers (e.g., 10.0)
3. **Click "Analyze Towers"** to process the data
4. **View results**:
   - Total tower count
   - OpenCellID tower count
   - FCC tower count
   - Tower density per square kilometer
   - **Total signal samples collected**
   - **Count of towers with >100 samples**
   - Average samples per tower
   - Interactive map showing tower locations
   - Signal analysis visualizations (histograms, scatter plots)
   - Radio type analysis
   - Detailed data tables

## 🔧 API Endpoints

### Health Check
- `GET /health` - Check API status and data loading

### Data Information
- `GET /data_info` - Get information about loaded datasets

### Tower Analysis
- `POST /analyze_towers` - Analyze towers within specified radius
  - **Request Body**: `{"latitude": float, "longitude": float, "radius": float}`
  - **Response**: Tower counts and detailed tower information

## 📈 Analysis Features

- **Haversine Distance Calculation**: Accurate geographic distance calculations
- **Radius-based Filtering**: Find all towers within the specified search radius
- **Dual Dataset Integration**: Combine analysis from both OpenCellID and FCC datasets
- **Signal Sample Analysis**: Count total samples and identify high-quality data sources
- **Density Calculations**: Calculate tower density per square kilometer
- **Interactive Visualization**: Map-based display with hover information
- **Statistical Analysis**: Sample distribution, radio type breakdown, and distance correlations

## 🎨 UI Features

- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Updates**: Instant feedback on API status and data loading
- **Interactive Maps**: Plotly-based maps with tower markers
- **Data Tables**: Sortable and filterable results tables
- **Progress Indicators**: Loading spinners and status messages

## 🔍 Example Use Cases

1. **Network Planning**: Analyze existing tower density for new infrastructure planning
2. **Signal Coverage**: Assess cellular signal coverage in specific areas
3. **Regulatory Compliance**: Review FCC-registered towers in target regions
4. **Market Analysis**: Understand infrastructure competition and gaps
5. **Emergency Planning**: Identify communication infrastructure for disaster response

## 🛡️ Error Handling

- **API Connection**: Graceful handling of backend connection issues
- **Data Validation**: Input validation for coordinates and radius
- **Missing Data**: Handling of incomplete or corrupted data records
- **Timeout Protection**: Configurable timeouts for large dataset processing

## 📝 Technical Notes

- **Data Loading**: Datasets are loaded once at startup for performance
- **Memory Management**: Efficient pandas operations for large datasets
- **CORS Support**: Cross-origin requests enabled for development
- **Async Processing**: FastAPI async endpoints for better performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and research purposes.

## 🆘 Troubleshooting

### Common Issues

1. **Backend not starting**: Check if data files exist in the `data/` folder
2. **Frontend can't connect**: Ensure the backend is running on port 8000
3. **Slow performance**: Large datasets may take time to load initially
4. **Memory issues**: Consider using smaller datasets for testing

### Performance Tips

- Use smaller radius values for faster analysis
- The initial data loading may take time with large datasets
- Consider data preprocessing for very large datasets

## 📞 Support

For issues or questions, please check the troubleshooting section above or create an issue in the project repository. 