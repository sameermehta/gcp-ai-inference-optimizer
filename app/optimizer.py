import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from .gcp_data import get_gcp_instances, get_instance_performance, calculate_cost_optimization

class EnterpriseOptimizer:
    """Advanced optimizer for enterprise AI inference workloads."""
    
    def __init__(self):
        self.performance_weight = 0.4
        self.cost_weight = 0.3
        self.reliability_weight = 0.2
        self.scalability_weight = 0.1
    
    def recommend_instance(self, 
                          model_size_gb: float,
                          target_qps: int,
                          target_latency_ms: int,
                          require_gpu: bool = False,
                          budget_constraint: Optional[float] = None,
                          region: str = "us-central1",
                          sla_requirement: str = "standard") -> Dict:
        """
        Advanced instance recommendation with enterprise features.
        
        Args:
            model_size_gb: Size of the AI model in GB
            target_qps: Target queries per second
            target_latency_ms: Maximum acceptable latency in milliseconds
            require_gpu: Whether GPU is required
            budget_constraint: Maximum hourly budget
            region: GCP region
            sla_requirement: SLA level (standard, premium, enterprise)
        """
        
        df = get_gcp_instances(region)
        
        # Filter instances based on requirements
        candidates = self._filter_candidates(df, model_size_gb, require_gpu, budget_constraint)
        
        if candidates.empty:
            return {"error": "No suitable instances found for the given requirements."}
        
        # Score candidates based on multiple criteria
        scored_candidates = self._score_candidates(candidates, target_qps, target_latency_ms, sla_requirement)
        
        # Get top recommendations
        top_recommendations = scored_candidates.head(3)
        
        # Calculate cost optimization
        cost_optimization = calculate_cost_optimization("", target_qps, target_latency_ms)
        
        return {
            "primary_recommendation": top_recommendations.iloc[0].to_dict(),
            "alternative_recommendations": top_recommendations.iloc[1:].to_dict('records'),
            "cost_optimization": cost_optimization,
            "performance_analysis": self._analyze_performance(top_recommendations.iloc[0], target_qps, target_latency_ms),
            "sla_compliance": self._check_sla_compliance(top_recommendations.iloc[0], sla_requirement)
        }
    
    def _filter_candidates(self, df: pd.DataFrame, model_size_gb: float, 
                          require_gpu: bool, budget_constraint: Optional[float]) -> pd.DataFrame:
        """Filter instances based on basic requirements."""
        
        # Filter for GPU if required
        if require_gpu:
            df = df[df['gpu'] > 0]
        else:
            df = df[df['gpu'] == 0]
        
        # Filter by memory (model size + buffer)
        required_memory = model_size_gb * 2.5  # 2.5x buffer for enterprise workloads
        df = df[df['memory_gb'] >= required_memory]
        
        # Filter by budget if specified
        if budget_constraint:
            df = df[df['price_per_hour'] <= budget_constraint]
        
        return df
    
    def _score_candidates(self, df: pd.DataFrame, target_qps: int, 
                         target_latency_ms: int, sla_requirement: str) -> pd.DataFrame:
        """Score candidates based on multiple criteria."""
        
        scores = []
        
        for _, instance in df.iterrows():
            perf = get_instance_performance(instance['name'])
            
            if not perf:
                continue
            
            # Performance score (0-1)
            qps_score = min(perf['throughput_qps'] / target_qps, 1.0)
            latency_score = max(0, 1 - (perf['inference_latency_ms'] / target_latency_ms))
            performance_score = (qps_score + latency_score) / 2
            
            # Cost efficiency score (0-1)
            cost_per_qps = instance['price_per_hour'] / perf['throughput_qps']
            cost_score = 1 / (1 + cost_per_qps * 100)  # Normalize
            
            # Reliability score based on instance family
            reliability_score = self._get_reliability_score(instance['family'])
            
            # Scalability score
            scalability_score = self._get_scalability_score(instance)
            
            # SLA compliance score
            sla_score = self._get_sla_score(instance, sla_requirement)
            
            # Weighted total score
            total_score = (
                self.performance_weight * performance_score +
                self.cost_weight * cost_score +
                self.reliability_weight * reliability_score +
                self.scalability_weight * scalability_score
            ) * sla_score
            
            scores.append({
                'instance_name': instance['name'],
                'performance_score': performance_score,
                'cost_score': cost_score,
                'reliability_score': reliability_score,
                'scalability_score': scalability_score,
                'sla_score': sla_score,
                'total_score': total_score,
                'monthly_cost': instance['price_per_hour'] * 24 * 30,
                'estimated_latency': perf['inference_latency_ms'],
                'estimated_throughput': perf['throughput_qps'],
                **instance.to_dict()
            })
        
        return pd.DataFrame(scores).sort_values('total_score', ascending=False)
    
    def _get_reliability_score(self, family: str) -> float:
        """Get reliability score based on instance family."""
        reliability_scores = {
            'e2': 0.85,  # General purpose, good reliability
            'c2': 0.90,  # Compute optimized, high reliability
            'n1': 0.95,  # Legacy, very reliable
            'a2': 0.98   # Latest generation, highest reliability
        }
        return reliability_scores.get(family, 0.80)
    
    def _get_scalability_score(self, instance: pd.Series) -> float:
        """Get scalability score based on instance specs."""
        # Higher CPU/memory ratios indicate better scalability
        cpu_memory_ratio = instance['cpu'] / instance['memory_gb']
        network_score = float(instance['network_performance'].split()[0]) / 16  # Normalize to 16 Gbps
        
        return (cpu_memory_ratio * 0.6 + network_score * 0.4)
    
    def _get_sla_score(self, instance: pd.Series, sla_requirement: str) -> float:
        """Get SLA compliance score."""
        if sla_requirement == "enterprise":
            # Enterprise SLA requires high-end instances
            if instance['family'] in ['a2', 'c2'] and instance['cpu'] >= 8:
                return 1.0
            else:
                return 0.7
        elif sla_requirement == "premium":
            # Premium SLA requires mid to high-end instances
            if instance['cpu'] >= 4:
                return 1.0
            else:
                return 0.8
        else:  # standard
            return 1.0
    
    def _analyze_performance(self, recommendation: pd.Series, target_qps: int, target_latency_ms: int) -> Dict:
        """Analyze performance characteristics of the recommended instance."""
        
        return {
            "qps_capacity": recommendation['estimated_throughput'],
            "qps_utilization": (target_qps / recommendation['estimated_throughput']) * 100,
            "latency_performance": recommendation['estimated_latency'],
            "latency_margin": target_latency_ms - recommendation['estimated_latency'],
            "performance_headroom": max(0, (recommendation['estimated_throughput'] - target_qps) / target_qps * 100),
            "risk_level": self._assess_risk_level(recommendation, target_qps, target_latency_ms)
        }
    
    def _assess_risk_level(self, instance: pd.Series, target_qps: int, target_latency_ms: int) -> str:
        """Assess risk level based on performance margins."""
        
        qps_margin = (instance['estimated_throughput'] - target_qps) / target_qps
        latency_margin = (target_latency_ms - instance['estimated_latency']) / target_latency_ms
        
        if qps_margin < 0.2 or latency_margin < 0.2:
            return "HIGH"
        elif qps_margin < 0.5 or latency_margin < 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _check_sla_compliance(self, instance: pd.Series, sla_requirement: str) -> Dict:
        """Check SLA compliance for the recommended instance."""
        
        sla_requirements = {
            "standard": {"uptime": 0.99, "support": "basic"},
            "premium": {"uptime": 0.995, "support": "enhanced"},
            "enterprise": {"uptime": 0.999, "support": "premium"}
        }
        
        sla_spec = sla_requirements.get(sla_requirement, sla_requirements["standard"])
        
        return {
            "sla_level": sla_requirement,
            "uptime_requirement": sla_spec["uptime"],
            "support_level": sla_spec["support"],
            "compliance_score": instance['sla_score'],
            "recommended_actions": self._get_sla_actions(instance, sla_requirement)
        }
    
    def _get_sla_actions(self, instance: pd.Series, sla_requirement: str) -> List[str]:
        """Get recommended actions for SLA compliance."""
        actions = []
        
        if sla_requirement == "enterprise" and instance['family'] not in ['a2', 'c2']:
            actions.append("Consider upgrading to A2 or C2 instance family for enterprise SLA")
        
        if instance['cpu'] < 8 and sla_requirement in ["premium", "enterprise"]:
            actions.append("Consider higher CPU instance for better performance guarantees")
        
        if instance['reliability_score'] < 0.9:
            actions.append("Implement redundancy and failover mechanisms")
        
        return actions

# Legacy function for backward compatibility
def recommend_instance(model_size_gb, qps, latency_ms, require_gpu=False):
    """Legacy recommendation function for backward compatibility."""
    optimizer = EnterpriseOptimizer()
    result = optimizer.recommend_instance(model_size_gb, qps, latency_ms, require_gpu)
    
    if "error" in result:
        return None
    
    primary = result["primary_recommendation"]
    return {
        "name": primary["instance_name"],
        "cpu": primary["cpu"],
        "memory_gb": primary["memory_gb"],
        "gpu": primary["gpu"],
        "price_per_hour": primary["price_per_hour"]
    }
