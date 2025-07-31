# GCP AI Inference Optimizer

This application recommends the most cost-effective Google Cloud Platform (GCP) compute instances for AI inference workloads, while maintaining required performance.

## Features
- User input for model specs (framework, size, QPS, latency requirement)
- GCP instance and pricing data
- Optimizer to match workloads to instance types
- Simple dashboard for recommendations (Streamlit)

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the dashboard:
   ```bash
   streamlit run app/dashboard.py
   ```

## Project Structure
- `app/optimizer.py`: Core logic for recommendations
- `app/gcp_data.py`: Fetch/store GCP instance/pricing data
- `app/dashboard.py`: Streamlit dashboard
