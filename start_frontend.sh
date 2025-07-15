#!/bin/bash

# Start the Allo Towers Frontend
echo "🎨 Starting Allo Towers Frontend..."
echo "📍 Working directory: $(pwd)"
echo "🌐 Opening Streamlit application..."

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate allo-towers

# Start the Streamlit app
cd frontend
streamlit run app.py 