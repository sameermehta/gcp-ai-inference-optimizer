import pandas as pd
import requests
import os
from typing import Dict, List, Optional
import json
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced GCP instance data with real-world specifications
GCP_INSTANCES = [
    # General Purpose
    {"name": "e2-standard-2", "family": "e2", "cpu": 2, "memory_gb": 8, "gpu": 0, "price_per_hour": 0.067, "region": "us-central1", "network_performance": "1 Gbps"},
    {"name": "e2-standard-4", "family": "e2", "cpu": 4, "memory_gb": 16, "gpu": 0, "price_per_hour": 0.134, "region": "us-central1", "network_performance": "1 Gbps"},
    {"name": "e2-standard-8", "family": "e2", "cpu": 8, "memory_gb": 32, "gpu": 0, "price_per_hour": 0.268, "region": "us-central1", "network_performance": "2 Gbps"},
    {"name": "e2-standard-16", "family": "e2", "cpu": 16, "memory_gb": 64, "gpu": 0, "price_per_hour": 0.536, "region": "us-central1", "network_performance": "4 Gbps"},
    
    # Memory Optimized
    {"name": "e2-highmem-2", "family": "e2", "cpu": 2, "memory_gb": 16, "gpu": 0, "price_per_hour": 0.089, "region": "us-central1", "network_performance": "1 Gbps"},
    {"name": "e2-highmem-4", "family": "e2", "cpu": 4, "memory_gb": 32, "gpu": 0, "price_per_hour": 0.178, "region": "us-central1", "network_performance": "1 Gbps"},
    {"name": "e2-highmem-8", "family": "e2", "cpu": 8, "memory_gb": 64, "gpu": 0, "price_per_hour": 0.356, "region": "us-central1", "network_performance": "2 Gbps"},
    
    # Compute Optimized
    {"name": "c2-standard-4", "family": "c2", "cpu": 4, "memory_gb": 16, "gpu": 0, "price_per_hour": 0.208, "region": "us-central1", "network_performance": "2 Gbps"},
    {"name": "c2-standard-8", "family": "c2", "cpu": 8, "memory_gb": 32, "gpu": 0, "price_per_hour": 0.416, "region": "us-central1", "network_performance": "4 Gbps"},
    {"name": "c2-standard-16", "family": "c2", "cpu": 16, "memory_gb": 64, "gpu": 0, "price_per_hour": 0.832, "region": "us-central1", "network_performance": "8 Gbps"},
    {"name": "c2-standard-30", "family": "c2", "cpu": 30, "memory_gb": 120, "gpu": 0, "price_per_hour": 1.560, "region": "us-central1", "network_performance": "16 Gbps"},
    
    # GPU Instances
    {"name": "n1-standard-4 + 1xT4", "family": "n1", "cpu": 4, "memory_gb": 15, "gpu": 1, "gpu_type": "T4", "price_per_hour": 0.472, "region": "us-central1", "network_performance": "1 Gbps"},
    {"name": "n1-standard-8 + 1xT4", "family": "n1", "cpu": 8, "memory_gb": 30, "gpu": 1, "gpu_type": "T4", "price_per_hour": 0.662, "region": "us-central1", "network_performance": "2 Gbps"},
    {"name": "n1-standard-16 + 1xT4", "family": "n1", "cpu": 16, "memory_gb": 60, "gpu": 1, "gpu_type": "T4", "price_per_hour": 1.042, "region": "us-central1", "network_performance": "4 Gbps"},
    {"name": "n1-standard-4 + 1xV100", "family": "n1", "cpu": 4, "memory_gb": 15, "gpu": 1, "gpu_type": "V100", "price_per_hour": 2.295, "region": "us-central1", "network_performance": "1 Gbps"},
    {"name": "n1-standard-8 + 1xV100", "family": "n1", "cpu": 8, "memory_gb": 30, "gpu": 1, "gpu_type": "V100", "price_per_hour": 2.485, "region": "us-central1", "network_performance": "2 Gbps"},
    {"name": "n1-standard-16 + 1xV100", "family": "n1", "cpu": 16, "memory_gb": 60, "gpu": 1, "gpu_type": "V100", "price_per_hour": 2.865, "region": "us-central1", "network_performance": "4 Gbps"},
    
    # A2 GPU Instances (for high-performance AI)
    {"name": "a2-standard-4", "family": "a2", "cpu": 4, "memory_gb": 16, "gpu": 1, "gpu_type": "A100", "price_per_hour": 1.212, "region": "us-central1", "network_performance": "2 Gbps"},
    {"name": "a2-standard-8", "family": "a2", "cpu": 8, "memory_gb": 32, "gpu": 1, "gpu_type": "A100", "price_per_hour": 1.412, "region": "us-central1", "network_performance": "4 Gbps"},
    {"name": "a2-standard-16", "family": "a2", "cpu": 16, "memory_gb": 64, "gpu": 1, "gpu_type": "A100", "price_per_hour": 1.812, "region": "us-central1", "network_performance": "8 Gbps"},
    {"name": "a2-highgpu-1g", "family": "a2", "cpu": 12, "memory_gb": 85, "gpu": 1, "gpu_type": "A100", "price_per_hour": 1.212, "region": "us-central1", "network_performance": "4 Gbps"},
    {"name": "a2-highgpu-4g", "family": "a2", "cpu": 48, "memory_gb": 340, "gpu": 4, "gpu_type": "A100", "price_per_hour": 4.848, "region": "us-central1", "network_performance": "16 Gbps"},
    {"name": "a2-highgpu-8g", "family": "a2", "cpu": 96, "memory_gb": 680, "gpu": 8, "gpu_type": "A100", "price_per_hour": 9.696, "region": "us-central1", "network_performance": "32 Gbps"},
]

# Performance benchmarks (simulated data for enterprise use)
PERFORMANCE_BENCHMARKS = {
    "e2-standard-4": {"inference_latency_ms": 45, "throughput_qps": 120, "memory_efficiency": 0.85},
    "e2-standard-8": {"inference_latency_ms": 32, "throughput_qps": 240, "memory_efficiency": 0.88},
    "c2-standard-8": {"inference_latency_ms": 28, "throughput_qps": 280, "memory_efficiency": 0.92},
    "n1-standard-8 + 1xT4": {"inference_latency_ms": 15, "throughput_qps": 450, "memory_efficiency": 0.95},
    "n1-standard-8 + 1xV100": {"inference_latency_ms": 8, "throughput_qps": 800, "memory_efficiency": 0.98},
    "a2-standard-8": {"inference_latency_ms": 5, "throughput_qps": 1200, "memory_efficiency": 0.99},
}

def get_gcp_instances(region: str = "us-central1") -> pd.DataFrame:
    """Return a DataFrame of available GCP instances for the specified region."""
    df = pd.DataFrame(GCP_INSTANCES)
    df = df[df['region'] == region]
    return df

def get_instance_performance(instance_name: str) -> Optional[Dict]:
    """Get performance benchmarks for a specific instance."""
    return PERFORMANCE_BENCHMARKS.get(instance_name)

def get_regions() -> List[str]:
    """Get available GCP regions."""
    return ["us-central1", "us-east1", "us-west1", "europe-west1", "asia-east1"]

def calculate_cost_optimization(current_instance: str, target_qps: int, target_latency: int) -> Dict:
    """Calculate cost optimization recommendations."""
    df = get_gcp_instances()
    
    # Find instances that meet requirements
    suitable_instances = []
    for _, instance in df.iterrows():
        perf = get_instance_performance(instance['name'])
        if perf and perf['throughput_qps'] >= target_qps and perf['inference_latency_ms'] <= target_latency:
            suitable_instances.append({
                'instance': instance['name'],
                'cost_per_hour': instance['price_per_hour'],
                'monthly_cost': instance['price_per_hour'] * 24 * 30,
                'performance_score': perf['throughput_qps'] / instance['price_per_hour'],
                'latency': perf['inference_latency_ms'],
                'throughput': perf['throughput_qps']
            })
    
    if suitable_instances:
        # Sort by cost efficiency
        suitable_instances.sort(key=lambda x: x['performance_score'], reverse=True)
        return {
            'recommendations': suitable_instances[:3],
            'best_value': suitable_instances[0],
            'cost_savings': None  # Would calculate based on current instance
        }
    
    return None

def get_historical_costs(instance_name: str, days: int = 30) -> List[Dict]:
    """Get historical cost data for an instance (simulated)."""
    base_cost = next((inst['price_per_hour'] for inst in GCP_INSTANCES if inst['name'] == instance_name), 0)
    
    historical_data = []
    for i in range(days):
        date = datetime.now() - timedelta(days=i)
        # Simulate some cost variation
        variation = 1 + (i % 7 - 3) * 0.05  # Weekly pattern
        historical_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'cost': base_cost * 24 * variation,
            'usage_hours': 24
        })
    
    return historical_data

def get_instance_utilization(instance_name: str) -> Dict:
    """Get current utilization metrics for an instance (simulated)."""
    return {
        'cpu_utilization': 75.5,
        'memory_utilization': 68.2,
        'gpu_utilization': 85.1 if 'gpu' in instance_name.lower() else 0,
        'network_utilization': 45.3,
        'disk_utilization': 32.8
    }
