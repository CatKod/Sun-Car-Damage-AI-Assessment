#!/bin/bash
# Script to run the Vehicle Damage Detection Streamlit App
echo "Starting Vehicle Damage Detection AI..."
echo ""
echo "Open your browser and go to: http://localhost:8501"
echo ""
streamlit run app/streamlit_app.py --server.port 8501
