#!/bin/bash

echo "Starting deployment..."

cd /home/sm3906/allo-towers

# Discard all local changes
git reset --hard HEAD
git clean -fd

echo "Pulling latest code from GitHub..."
git pull origin main

echo "Activating virtual environment..."
source /home/sm3906/venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Restarting FastAPI backend service..."
sudo systemctl restart allo-backend.service

echo "Restarting Streamlit frontend service..."
sudo systemctl restart allo-frontend.service

echo "Deployment complete."
