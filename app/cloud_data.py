import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Multi-cloud instance data with real-world specifications
MULTI_CLOUD_INSTANCES = {
    "AWS": [
        # General Purpose
        {"name": "t3.medium", "family": "t3", "cpu": 2, "memory_gb": 4, "gpu": 0, "price_per_hour": 0.0416, "region": "us-east-1", "network_performance": "Up to 5 Gbps"},
        {"name": "m5.large", "family": "m5", "cpu": 2, "memory_gb": 8, "gpu": 0, "price_per_hour": 0.096, "region": "us-east-1", "network_performance": "Up to 10 Gbps"},
        {"name": "m5.xlarge", "family": "m5", "cpu": 4, "memory_gb": 16, "gpu": 0, "price_per_hour": 0.192, "region": "us-east-1", "network_performance": "Up to 10 Gbps"},
        {"name": "m5.2xlarge", "family": "m5", "cpu": 8, "memory_gb": 32, "gpu": 0, "price_per_hour": 0.384, "region": "us-east-1", "network_performance": "Up to 10 Gbps"},
        
        # Compute Optimized
        {"name": "c5.large", "family": "c5", "cpu": 2, "memory_gb": 4, "gpu": 0, "price_per_hour": 0.085, "region": "us-east-1", "network_performance": "Up to 10 Gbps"},
        {"name": "c5.xlarge", "family": "c5", "cpu": 4, "memory_gb": 8, "gpu": 0, "price_per_hour": 0.17, "region": "us-east-1", "network_performance": "Up to 10 Gbps"},
        {"name": "c5.2xlarge", "family": "c5", "cpu": 8, "memory_gb": 16, "gpu": 0, "price_per_hour": 0.34, "region": "us-east-1", "network_performance": "Up to 10 Gbps"},
        
        # Memory Optimized
        {"name": "r5.large", "family": "r5", "cpu": 2, "memory_gb": 16, "gpu": 0, "price_per_hour": 0.126, "region": "us-east-1", "network_performance": "Up to 10 Gbps"},
        {"name": "r5.xlarge", "family": "r5", "cpu": 4, "memory_gb": 32, "gpu": 0, "price_per_hour": 0.252, "region": "us-east-1", "network_performance": "Up to 10 Gbps"},
        {"name": "r5.2xlarge", "family": "r5", "cpu": 8, "memory_gb": 64, "gpu": 0, "price_per_hour": 0.504, "region": "us-east-1", "network_performance": "Up to 10 Gbps"},
        
        # GPU Instances
        {"name": "g4dn.xlarge", "family": "g4dn", "cpu": 4, "memory_gb": 16, "gpu": 1, "gpu_type": "T4", "price_per_hour": 0.526, "region": "us-east-1", "network_performance": "Up to 25 Gbps"},
        {"name": "g4dn.2xlarge", "family": "g4dn", "cpu": 8, "memory_gb": 32, "gpu": 1, "gpu_type": "T4", "price_per_hour": 0.752, "region": "us-east-1", "network_performance": "Up to 25 Gbps"},
        {"name": "p3.2xlarge", "family": "p3", "cpu": 8, "memory_gb": 61, "gpu": 1, "gpu_type": "V100", "price_per_hour": 3.06, "region": "us-east-1", "network_performance": "Up to 10 Gbps"},
        {"name": "p3.8xlarge", "family": "p3", "cpu": 32, "memory_gb": 244, "gpu": 4, "gpu_type": "V100", "price_per_hour": 12.24, "region": "us-east-1", "network_performance": "Up to 10 Gbps"},
    ],
    
    "Azure": [
        # General Purpose
        {"name": "Standard_B2s", "family": "B", "cpu": 2, "memory_gb": 4, "gpu": 0, "price_per_hour": 0.0416, "region": "eastus", "network_performance": "Up to 4 Gbps"},
        {"name": "Standard_D2s_v3", "family": "D", "cpu": 2, "memory_gb": 8, "gpu": 0, "price_per_hour": 0.096, "region": "eastus", "network_performance": "Up to 6 Gbps"},
        {"name": "Standard_D4s_v3", "family": "D", "cpu": 4, "memory_gb": 16, "gpu": 0, "price_per_hour": 0.192, "region": "eastus", "network_performance": "Up to 12 Gbps"},
        {"name": "Standard_D8s_v3", "family": "D", "cpu": 8, "memory_gb": 32, "gpu": 0, "price_per_hour": 0.384, "region": "eastus", "network_performance": "Up to 16 Gbps"},
        
        # Compute Optimized
        {"name": "Standard_F2s_v2", "family": "F", "cpu": 2, "memory_gb": 4, "gpu": 0, "price_per_hour": 0.085, "region": "eastus", "network_performance": "Up to 4 Gbps"},
        {"name": "Standard_F4s_v2", "family": "F", "cpu": 4, "memory_gb": 8, "gpu": 0, "price_per_hour": 0.17, "region": "eastus", "network_performance": "Up to 8 Gbps"},
        {"name": "Standard_F8s_v2", "family": "F", "cpu": 8, "memory_gb": 16, "gpu": 0, "price_per_hour": 0.34, "region": "eastus", "network_performance": "Up to 16 Gbps"},
        
        # Memory Optimized
        {"name": "Standard_E2s_v3", "family": "E", "cpu": 2, "memory_gb": 16, "gpu": 0, "price_per_hour": 0.126, "region": "eastus", "network_performance": "Up to 4 Gbps"},
        {"name": "Standard_E4s_v3", "family": "E", "cpu": 4, "memory_gb": 32, "gpu": 0, "price_per_hour": 0.252, "region": "eastus", "network_performance": "Up to 8 Gbps"},
        {"name": "Standard_E8s_v3", "family": "E", "cpu": 8, "memory_gb": 64, "gpu": 0, "price_per_hour": 0.504, "region": "eastus", "network_performance": "Up to 16 Gbps"},
        
        # GPU Instances
        {"name": "Standard_NC4as_T4_v3", "family": "NC", "cpu": 4, "memory_gb": 28, "gpu": 1, "gpu_type": "T4", "price_per_hour": 0.526, "region": "eastus", "network_performance": "Up to 8 Gbps"},
        {"name": "Standard_NC8as_T4_v3", "family": "NC", "cpu": 8, "memory_gb": 56, "gpu": 1, "gpu_type": "T4", "price_per_hour": 0.752, "region": "eastus", "network_performance": "Up to 16 Gbps"},
        {"name": "Standard_NC6s_v3", "family": "NC", "cpu": 6, "memory_gb": 112, "gpu": 1, "gpu_type": "V100", "price_per_hour": 3.06, "region": "eastus", "network_performance": "Up to 24 Gbps"},
        {"name": "Standard_NC24rs_v3", "family": "NC", "cpu": 24, "memory_gb": 448, "gpu": 4, "gpu_type": "V100", "price_per_hour": 12.24, "region": "eastus", "network_performance": "Up to 32 Gbps"},
    ],
    
    "GCP": [
        # General Purpose
        {"name": "e2-standard-2", "family": "e2", "cpu": 2, "memory_gb": 8, "gpu": 0, "price_per_hour": 0.067, "region": "us-central1", "network_performance": "1 Gbps"},
        {"name": "e2-standard-4", "family": "e2", "cpu": 4, "memory_gb": 16, "gpu": 0, "price_per_hour": 0.134, "region": "us-central1", "network_performance": "1 Gbps"},
        {"name": "e2-standard-8", "family": "e2", "cpu": 8, "memory_gb": 32, "gpu": 0, "price_per_hour": 0.268, "region": "us-central1", "network_performance": "2 Gbps"},
        
        # Compute Optimized
        {"name": "c2-standard-4", "family": "c2", "cpu": 4, "memory_gb": 16, "gpu": 0, "price_per_hour": 0.208, "region": "us-central1", "network_performance": "2 Gbps"},
        {"name": "c2-standard-8", "family": "c2", "cpu": 8, "memory_gb": 32, "gpu": 0, "price_per_hour": 0.416, "region": "us-central1", "network_performance": "4 Gbps"},
        {"name": "c2-standard-16", "family": "c2", "cpu": 16, "memory_gb": 64, "gpu": 0, "price_per_hour": 0.832, "region": "us-central1", "network_performance": "8 Gbps"},
        
        # Memory Optimized
        {"name": "e2-highmem-2", "family": "e2", "cpu": 2, "memory_gb": 16, "gpu": 0, "price_per_hour": 0.089, "region": "us-central1", "network_performance": "1 Gbps"},
        {"name": "e2-highmem-4", "family": "e2", "cpu": 4, "memory_gb": 32, "gpu": 0, "price_per_hour": 0.178, "region": "us-central1", "network_performance": "1 Gbps"},
        {"name": "e2-highmem-8", "family": "e2", "cpu": 8, "memory_gb": 64, "gpu": 0, "price_per_hour": 0.356, "region": "us-central1", "network_performance": "2 Gbps"},
        
        # GPU Instances
        {"name": "n1-standard-4 + 1xT4", "family": "n1", "cpu": 4, "memory_gb": 15, "gpu": 1, "gpu_type": "T4", "price_per_hour": 0.472, "region": "us-central1", "network_performance": "1 Gbps"},
        {"name": "n1-standard-8 + 1xT4", "family": "n1", "cpu": 8, "memory_gb": 30, "gpu": 1, "gpu_type": "T4", "price_per_hour": 0.662, "region": "us-central1", "network_performance": "2 Gbps"},
        {"name": "n1-standard-8 + 1xV100", "family": "n1", "cpu": 8, "memory_gb": 30, "gpu": 1, "gpu_type": "V100", "price_per_hour": 2.485, "region": "us-central1", "network_performance": "2 Gbps"},
        {"name": "a2-standard-8", "family": "a2", "cpu": 8, "memory_gb": 32, "gpu": 1, "gpu_type": "A100", "price_per_hour": 1.412, "region": "us-central1", "network_performance": "4 Gbps"},
    ],
    
    "Oracle": [
        # General Purpose
        {"name": "VM.Standard2.1", "family": "Standard2", "cpu": 1, "memory_gb": 6, "gpu": 0, "price_per_hour": 0.05, "region": "us-ashburn-1", "network_performance": "1 Gbps"},
        {"name": "VM.Standard2.2", "family": "Standard2", "cpu": 2, "memory_gb": 12, "gpu": 0, "price_per_hour": 0.1, "region": "us-ashburn-1", "network_performance": "1 Gbps"},
        {"name": "VM.Standard2.4", "family": "Standard2", "cpu": 4, "memory_gb": 24, "gpu": 0, "price_per_hour": 0.2, "region": "us-ashburn-1", "network_performance": "2 Gbps"},
        {"name": "VM.Standard2.8", "family": "Standard2", "cpu": 8, "memory_gb": 48, "gpu": 0, "price_per_hour": 0.4, "region": "us-ashburn-1", "network_performance": "4 Gbps"},
        
        # Compute Optimized
        {"name": "VM.Standard3.Flex", "family": "Standard3", "cpu": 2, "memory_gb": 8, "gpu": 0, "price_per_hour": 0.085, "region": "us-ashburn-1", "network_performance": "2 Gbps"},
        {"name": "VM.Standard3.Flex", "family": "Standard3", "cpu": 4, "memory_gb": 16, "gpu": 0, "price_per_hour": 0.17, "region": "us-ashburn-1", "network_performance": "4 Gbps"},
        {"name": "VM.Standard3.Flex", "family": "Standard3", "cpu": 8, "memory_gb": 32, "gpu": 0, "price_per_hour": 0.34, "region": "us-ashburn-1", "network_performance": "8 Gbps"},
        
        # Memory Optimized
        {"name": "VM.Standard2.1", "family": "Standard2", "cpu": 1, "memory_gb": 12, "gpu": 0, "price_per_hour": 0.075, "region": "us-ashburn-1", "network_performance": "1 Gbps"},
        {"name": "VM.Standard2.2", "family": "Standard2", "cpu": 2, "memory_gb": 24, "gpu": 0, "price_per_hour": 0.15, "region": "us-ashburn-1", "network_performance": "1 Gbps"},
        {"name": "VM.Standard2.4", "family": "Standard2", "cpu": 4, "memory_gb": 48, "gpu": 0, "price_per_hour": 0.3, "region": "us-ashburn-1", "network_performance": "2 Gbps"},
        
        # GPU Instances (Limited selection for Oracle)
        {"name": "BM.GPU3.8", "family": "BM", "cpu": 52, "memory_gb": 768, "gpu": 8, "gpu_type": "V100", "price_per_hour": 15.0, "region": "us-ashburn-1", "network_performance": "25 Gbps"},
    ]
}

# Performance benchmarks for each cloud provider
PERFORMANCE_BENCHMARKS = {
    "AWS": {
        "t3.medium": {"inference_latency_ms": 50, "throughput_qps": 80, "memory_efficiency": 0.82},
        "m5.large": {"inference_latency_ms": 40, "throughput_qps": 120, "memory_efficiency": 0.85},
        "m5.xlarge": {"inference_latency_ms": 35, "throughput_qps": 200, "memory_efficiency": 0.88},
        "c5.xlarge": {"inference_latency_ms": 30, "throughput_qps": 250, "memory_efficiency": 0.90},
        "g4dn.xlarge": {"inference_latency_ms": 12, "throughput_qps": 400, "memory_efficiency": 0.94},
        "p3.2xlarge": {"inference_latency_ms": 8, "throughput_qps": 800, "memory_efficiency": 0.97},
    },
    "Azure": {
        "Standard_D4s_v3": {"inference_latency_ms": 38, "throughput_qps": 180, "memory_efficiency": 0.86},
        "Standard_F4s_v2": {"inference_latency_ms": 32, "throughput_qps": 220, "memory_efficiency": 0.89},
        "Standard_E4s_v3": {"inference_latency_ms": 42, "throughput_qps": 160, "memory_efficiency": 0.84},
        "Standard_NC4as_T4_v3": {"inference_latency_ms": 15, "throughput_qps": 380, "memory_efficiency": 0.93},
        "Standard_NC6s_v3": {"inference_latency_ms": 9, "throughput_qps": 750, "memory_efficiency": 0.96},
    },
    "GCP": {
        "e2-standard-4": {"inference_latency_ms": 45, "throughput_qps": 120, "memory_efficiency": 0.85},
        "c2-standard-8": {"inference_latency_ms": 28, "throughput_qps": 280, "memory_efficiency": 0.92},
        "n1-standard-8 + 1xT4": {"inference_latency_ms": 15, "throughput_qps": 450, "memory_efficiency": 0.95},
        "a2-standard-8": {"inference_latency_ms": 5, "throughput_qps": 1200, "memory_efficiency": 0.99},
    },
    "Oracle": {
        "VM.Standard2.4": {"inference_latency_ms": 48, "throughput_qps": 100, "memory_efficiency": 0.80},
        "VM.Standard3.Flex": {"inference_latency_ms": 35, "throughput_qps": 180, "memory_efficiency": 0.85},
    }
}

# Cloud provider regions
CLOUD_REGIONS = {
    "AWS": ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
    "Azure": ["eastus", "westus2", "westeurope", "southeastasia"],
    "GCP": ["us-central1", "us-east1", "europe-west1", "asia-east1"],
    "Oracle": ["us-ashburn-1", "us-phoenix-1", "eu-frankfurt-1", "ap-tokyo-1"]
}

# Cost optimization strategies
COST_OPTIMIZATION_STRATEGIES = {
    "Reserved Instances": {
        "AWS": {"savings": 0.60, "commitment": "1-3 years", "description": "Up to 60% savings with reserved instances"},
        "Azure": {"savings": 0.55, "commitment": "1-3 years", "description": "Up to 55% savings with reserved instances"},
        "GCP": {"savings": 0.55, "commitment": "1-3 years", "description": "Up to 55% savings with committed use discounts"},
        "Oracle": {"savings": 0.50, "commitment": "1-3 years", "description": "Up to 50% savings with universal credits"}
    },
    "Spot Instances": {
        "AWS": {"savings": 0.90, "availability": "Variable", "description": "Up to 90% savings but can be interrupted"},
        "Azure": {"savings": 0.85, "availability": "Variable", "description": "Up to 85% savings with low-priority VMs"},
        "GCP": {"savings": 0.80, "availability": "Variable", "description": "Up to 80% savings with preemptible instances"},
        "Oracle": {"savings": 0.70, "availability": "Limited", "description": "Limited spot instance availability"}
    },
    "Auto Scaling": {
        "AWS": {"savings": 0.40, "complexity": "Medium", "description": "Scale based on demand to reduce idle costs"},
        "Azure": {"savings": 0.35, "complexity": "Medium", "description": "Scale based on demand to reduce idle costs"},
        "GCP": {"savings": 0.45, "complexity": "Medium", "description": "Scale based on demand to reduce idle costs"},
        "Oracle": {"savings": 0.30, "complexity": "High", "description": "Limited auto-scaling capabilities"}
    },
    "Right Sizing": {
        "AWS": {"savings": 0.25, "effort": "Low", "description": "Match instance size to actual workload needs"},
        "Azure": {"savings": 0.25, "effort": "Low", "description": "Match instance size to actual workload needs"},
        "GCP": {"savings": 0.25, "effort": "Low", "description": "Match instance size to actual workload needs"},
        "Oracle": {"savings": 0.20, "effort": "Medium", "description": "Match instance size to actual workload needs"}
    }
}

def get_all_cloud_instances(region: str = None) -> pd.DataFrame:
    """Get instances from all cloud providers."""
    all_instances = []
    
    for cloud_provider, instances in MULTI_CLOUD_INSTANCES.items():
        for instance in instances:
            instance_data = instance.copy()
            instance_data['cloud_provider'] = cloud_provider
            if region is None or instance['region'] == region:
                all_instances.append(instance_data)
    
    return pd.DataFrame(all_instances)

def get_cloud_instances(cloud_provider: str, region: str = None) -> pd.DataFrame:
    """Get instances for a specific cloud provider."""
    instances = MULTI_CLOUD_INSTANCES.get(cloud_provider, [])
    
    if region:
        instances = [inst for inst in instances if inst['region'] == region]
    
    df = pd.DataFrame(instances)
    if not df.empty:
        df['cloud_provider'] = cloud_provider
    
    return df

def get_instance_performance(instance_name: str, cloud_provider: str) -> Optional[Dict]:
    """Get performance benchmarks for a specific instance."""
    benchmarks = PERFORMANCE_BENCHMARKS.get(cloud_provider, {})
    return benchmarks.get(instance_name)

def get_cloud_providers() -> List[str]:
    """Get available cloud providers."""
    return list(MULTI_CLOUD_INSTANCES.keys())

def get_cloud_regions(cloud_provider: str) -> List[str]:
    """Get available regions for a cloud provider."""
    return CLOUD_REGIONS.get(cloud_provider, [])

def calculate_multi_cloud_cost_optimization(target_qps: int, target_latency: int, budget_constraint: float = None) -> Dict:
    """Calculate cost optimization across all cloud providers."""
    all_instances = get_all_cloud_instances()
    
    suitable_instances = []
    
    for _, instance in all_instances.iterrows():
        perf = get_instance_performance(instance['name'], instance['cloud_provider'])
        
        if perf and perf['throughput_qps'] >= target_qps and perf['inference_latency_ms'] <= target_latency:
            if budget_constraint is None or instance['price_per_hour'] <= budget_constraint:
                suitable_instances.append({
                    'cloud_provider': instance['cloud_provider'],
                    'instance': instance['name'],
                    'cost_per_hour': instance['price_per_hour'],
                    'monthly_cost': instance['price_per_hour'] * 24 * 30,
                    'performance_score': perf['throughput_qps'] / instance['price_per_hour'],
                    'latency': perf['inference_latency_ms'],
                    'throughput': perf['throughput_qps'],
                    'memory_gb': instance['memory_gb'],
                    'cpu': instance['cpu'],
                    'gpu': instance['gpu']
                })
    
    if suitable_instances:
        # Sort by cost efficiency
        suitable_instances.sort(key=lambda x: x['performance_score'], reverse=True)
        
        # Group by cloud provider
        by_provider = {}
        for instance in suitable_instances:
            provider = instance['cloud_provider']
            if provider not in by_provider:
                by_provider[provider] = []
            by_provider[provider].append(instance)
        
        return {
            'recommendations': suitable_instances[:10],  # Top 10 overall
            'by_provider': by_provider,
            'best_value': suitable_instances[0],
            'cost_comparison': {
                provider: min(instances, key=lambda x: x['monthly_cost']) 
                for provider, instances in by_provider.items()
            }
        }
    
    return None

def get_cost_optimization_strategies(cloud_provider: str) -> Dict:
    """Get cost optimization strategies for a specific cloud provider."""
    strategies = {}
    for strategy_name, provider_data in COST_OPTIMIZATION_STRATEGIES.items():
        if cloud_provider in provider_data:
            strategies[strategy_name] = provider_data[cloud_provider]
    return strategies

def get_cloud_provider_comparison() -> pd.DataFrame:
    """Get a comparison table of cloud providers."""
    comparison_data = []
    
    for provider in get_cloud_providers():
        instances = get_cloud_instances(provider)
        if not instances.empty:
            avg_cost = instances['price_per_hour'].mean()
            min_cost = instances['price_per_hour'].min()
            max_cost = instances['price_per_hour'].max()
            
            comparison_data.append({
                'Cloud Provider': provider,
                'Avg Hourly Cost': avg_cost,
                'Min Hourly Cost': min_cost,
                'Max Hourly Cost': max_cost,
                'Instance Count': len(instances),
                'GPU Instances': len(instances[instances['gpu'] > 0]),
                'Regions Available': len(get_cloud_regions(provider))
            })
    
    return pd.DataFrame(comparison_data)

def get_historical_costs_multi_cloud(instance_name: str, cloud_provider: str, days: int = 30) -> List[Dict]:
    """Get historical cost data for an instance across cloud providers."""
    instances = get_cloud_instances(cloud_provider)
    base_cost = instances[instances['name'] == instance_name]['price_per_hour'].iloc[0] if not instances.empty else 0
    
    historical_data = []
    for i in range(days):
        date = datetime.now() - timedelta(days=i)
        # Simulate some cost variation
        variation = 1 + (i % 7 - 3) * 0.05  # Weekly pattern
        historical_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'cost': base_cost * 24 * variation,
            'usage_hours': 24,
            'cloud_provider': cloud_provider
        })
    
    return historical_data 