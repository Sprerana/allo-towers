#!/bin/bash

# Start the Allo Towers Backend API
echo "🚀 Starting Allo Towers Backend API..."
echo "📍 Working directory: $(pwd)"
echo "📊 Loading tower datasets..."

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate allo-towers

# Start the FastAPI server
cd backend
python main.py 