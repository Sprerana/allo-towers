#!/bin/bash

echo "Starting deployment..."

cd /home/sm3906/allo-towers

echo "Remove local changed of VM"
git reset --hard HEAD
git clean -fd

echo "Pulling latest code from GitHub..."
git pull origin main

echo "Activating virtual environment..."
source /home/sm3906/venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

#sudo systemctl stop allo-frontend

sudo systemctl restart allo-frontend

#sudo systemctl stop allo-frontend

sudo systemctl restart allo-backend

echo "Deployment complete."
