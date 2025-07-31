import pandas as pd

# For MVP, we'll use a static list. In production, fetch from GCP API or pricing files.
GCP_INSTANCES = [
    {"name": "n1-standard-4", "cpu": 4, "memory_gb": 15, "gpu": 0, "price_per_hour": 0.190},
    {"name": "n1-standard-8", "cpu": 8, "memory_gb": 30, "gpu": 0, "price_per_hour": 0.380},
    {"name": "n1-highmem-8", "cpu": 8, "memory_gb": 52, "gpu": 0, "price_per_hour": 0.418},
    {"name": "n1-standard-8 + 1xV100", "cpu": 8, "memory_gb": 52, "gpu": 1, "price_per_hour": 2.475},
    # Add more as needed
]

def get_gcp_instances():
    """Return a DataFrame of available GCP instances."""
    return pd.DataFrame(GCP_INSTANCES)
