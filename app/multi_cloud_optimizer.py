import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from cloud_data import (
    get_all_cloud_instances, get_cloud_instances, get_instance_performance,
    get_cloud_providers, get_cloud_regions, calculate_multi_cloud_cost_optimization,
    get_cost_optimization_strategies, get_cloud_provider_comparison
)

class MultiCloudOptimizer:
    """Advanced multi-cloud optimizer for enterprise AI inference workloads."""
    
    def __init__(self):
        self.performance_weight = 0.35
        self.cost_weight = 0.30
        self.reliability_weight = 0.20
        self.scalability_weight = 0.15
        
        # Cloud provider reliability scores (based on industry data)
        self.cloud_reliability_scores = {
            "AWS": 0.99,      # Industry leader
            "Azure": 0.98,    # Strong enterprise focus
            "GCP": 0.97,      # Good reliability
            "Oracle": 0.95    # Improving but newer
        }
        
        # Cloud provider cost competitiveness
        self.cost_competitiveness = {
            "AWS": 0.85,      # Premium pricing
            "Azure": 0.90,    # Competitive pricing
            "GCP": 0.95,      # Often most competitive
            "Oracle": 0.88    # Aggressive pricing
        }
    
    def recommend_multi_cloud(self, 
                             model_size_gb: float,
                             target_qps: int,
                             target_latency_ms: int,
                             require_gpu: bool = False,
                             budget_constraint: Optional[float] = None,
                             preferred_clouds: List[str] = None,
                             sla_requirement: str = "standard") -> Dict:
        """
        Multi-cloud instance recommendation with enterprise features.
        
        Args:
            model_size_gb: Size of the AI model in GB
            target_qps: Target queries per second
            target_latency_ms: Maximum acceptable latency in milliseconds
            require_gpu: Whether GPU is required
            budget_constraint: Maximum hourly budget
            preferred_clouds: List of preferred cloud providers
            sla_requirement: SLA level (standard, premium, enterprise)
        """
        
        # Get all cloud instances
        all_instances = get_all_cloud_instances()
        
        # Filter by preferred clouds if specified
        if preferred_clouds:
            all_instances = all_instances[all_instances['cloud_provider'].isin(preferred_clouds)]
        
        # Filter instances based on requirements
        candidates = self._filter_multi_cloud_candidates(
            all_instances, model_size_gb, require_gpu, budget_constraint
        )
        
        if candidates.empty:
            return {"error": "No suitable instances found across cloud providers."}
        
        # Score candidates based on multiple criteria
        scored_candidates = self._score_multi_cloud_candidates(
            candidates, target_qps, target_latency_ms, sla_requirement
        )
        
        # Get top recommendations by cloud provider
        top_recommendations = self._get_top_recommendations_by_provider(scored_candidates)
        
        # Calculate cost optimization across clouds
        cost_optimization = calculate_multi_cloud_cost_optimization(target_qps, target_latency_ms, budget_constraint)
        
        # Generate cloud provider comparison
        cloud_comparison = self._generate_cloud_comparison(scored_candidates)
        
        return {
            "primary_recommendation": top_recommendations["best_overall"],
            "recommendations_by_provider": top_recommendations["by_provider"],
            "cost_optimization": cost_optimization,
            "cloud_comparison": cloud_comparison,
            "performance_analysis": self._analyze_multi_cloud_performance(
                top_recommendations["best_overall"], target_qps, target_latency_ms
            ),
            "sla_compliance": self._check_multi_cloud_sla_compliance(
                top_recommendations["best_overall"], sla_requirement
            ),
            "cost_savings_opportunities": self._identify_cost_savings_opportunities(
                scored_candidates, target_qps, target_latency_ms
            )
        }
    
    def _filter_multi_cloud_candidates(self, df: pd.DataFrame, model_size_gb: float, 
                                     require_gpu: bool, budget_constraint: Optional[float]) -> pd.DataFrame:
        """Filter instances based on basic requirements across all clouds."""
        
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
    
    def _score_multi_cloud_candidates(self, df: pd.DataFrame, target_qps: int, 
                                    target_latency_ms: int, sla_requirement: str) -> pd.DataFrame:
        """Score candidates based on multiple criteria across all clouds."""
        
        scores = []
        
        for _, instance in df.iterrows():
            perf = get_instance_performance(instance['name'], instance['cloud_provider'])
            
            if not perf:
                continue
            
            # Performance score (0-1)
            qps_score = min(perf['throughput_qps'] / target_qps, 1.0)
            latency_score = max(0, 1 - (perf['inference_latency_ms'] / target_latency_ms))
            performance_score = (qps_score + latency_score) / 2
            
            # Cost efficiency score (0-1)
            cost_per_qps = instance['price_per_hour'] / perf['throughput_qps']
            cost_score = 1 / (1 + cost_per_qps * 100)  # Normalize
            
            # Cloud provider reliability score
            reliability_score = self.cloud_reliability_scores.get(instance['cloud_provider'], 0.80)
            
            # Scalability score
            scalability_score = self._get_multi_cloud_scalability_score(instance)
            
            # SLA compliance score
            sla_score = self._get_multi_cloud_sla_score(instance, sla_requirement)
            
            # Cost competitiveness bonus
            cost_competitiveness_bonus = self.cost_competitiveness.get(instance['cloud_provider'], 0.85)
            
            # Weighted total score
            total_score = (
                self.performance_weight * performance_score +
                self.cost_weight * cost_score * cost_competitiveness_bonus +
                self.reliability_weight * reliability_score +
                self.scalability_weight * scalability_score
            ) * sla_score
            
            scores.append({
                'cloud_provider': instance['cloud_provider'],
                'instance_name': instance['name'],
                'performance_score': performance_score,
                'cost_score': cost_score,
                'reliability_score': reliability_score,
                'scalability_score': scalability_score,
                'sla_score': sla_score,
                'cost_competitiveness': cost_competitiveness_bonus,
                'total_score': total_score,
                'monthly_cost': instance['price_per_hour'] * 24 * 30,
                'estimated_latency': perf['inference_latency_ms'],
                'estimated_throughput': perf['throughput_qps'],
                'cpu': instance['cpu'],
                'memory_gb': instance['memory_gb'],
                'gpu': instance['gpu'],
                'gpu_type': instance.get('gpu_type', 'None'),
                'network_performance': instance['network_performance'],
                'region': instance['region']
            })
        
        return pd.DataFrame(scores).sort_values('total_score', ascending=False)
    
    def _get_multi_cloud_scalability_score(self, instance: pd.Series) -> float:
        """Get scalability score based on instance specs and cloud provider."""
        # Higher CPU/memory ratios indicate better scalability
        cpu_memory_ratio = instance['cpu'] / instance['memory_gb']
        network_score = float(instance['network_performance'].split()[0]) / 16  # Normalize to 16 Gbps
        
        # Cloud provider scalability bonus
        scalability_bonus = {
            "AWS": 1.0,    # Excellent auto-scaling
            "Azure": 0.95, # Good auto-scaling
            "GCP": 0.98,   # Very good auto-scaling
            "Oracle": 0.85 # Limited auto-scaling
        }.get(instance['cloud_provider'], 0.80)
        
        return (cpu_memory_ratio * 0.6 + network_score * 0.4) * scalability_bonus
    
    def _get_multi_cloud_sla_score(self, instance: pd.Series, sla_requirement: str) -> float:
        """Get SLA compliance score across cloud providers."""
        if sla_requirement == "enterprise":
            # Enterprise SLA requires high-end instances and reliable clouds
            if instance['cloud_provider'] in ['AWS', 'Azure'] and instance['cpu'] >= 8:
                return 1.0
            elif instance['cloud_provider'] in ['GCP', 'Oracle'] and instance['cpu'] >= 8:
                return 0.9
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
    
    def _get_top_recommendations_by_provider(self, scored_candidates: pd.DataFrame) -> Dict:
        """Get top recommendations organized by cloud provider."""
        by_provider = {}
        best_overall = None
        best_score = 0
        
        for provider in get_cloud_providers():
            provider_instances = scored_candidates[scored_candidates['cloud_provider'] == provider]
            if not provider_instances.empty:
                by_provider[provider] = provider_instances.head(3).to_dict('records')
                
                # Track best overall
                top_provider_instance = provider_instances.iloc[0]
                if top_provider_instance['total_score'] > best_score:
                    best_score = top_provider_instance['total_score']
                    best_overall = top_provider_instance.to_dict()
        
        return {
            "best_overall": best_overall,
            "by_provider": by_provider
        }
    
    def _generate_cloud_comparison(self, scored_candidates: pd.DataFrame) -> Dict:
        """Generate comprehensive cloud provider comparison."""
        comparison = {}
        
        for provider in get_cloud_providers():
            provider_instances = scored_candidates[scored_candidates['cloud_provider'] == provider]
            if not provider_instances.empty:
                comparison[provider] = {
                    "avg_monthly_cost": provider_instances['monthly_cost'].mean(),
                    "min_monthly_cost": provider_instances['monthly_cost'].min(),
                    "max_monthly_cost": provider_instances['monthly_cost'].max(),
                    "avg_performance_score": provider_instances['performance_score'].mean(),
                    "avg_latency": provider_instances['estimated_latency'].mean(),
                    "avg_throughput": provider_instances['estimated_throughput'].mean(),
                    "instance_count": len(provider_instances),
                    "gpu_instances": len(provider_instances[provider_instances['gpu'] > 0]),
                    "reliability_score": self.cloud_reliability_scores.get(provider, 0.80),
                    "cost_competitiveness": self.cost_competitiveness.get(provider, 0.85)
                }
        
        return comparison
    
    def _analyze_multi_cloud_performance(self, recommendation: Dict, target_qps: int, target_latency_ms: int) -> Dict:
        """Analyze performance characteristics of the recommended instance."""
        
        return {
            "qps_capacity": recommendation['estimated_throughput'],
            "qps_utilization": (target_qps / recommendation['estimated_throughput']) * 100,
            "latency_performance": recommendation['estimated_latency'],
            "latency_margin": target_latency_ms - recommendation['estimated_latency'],
            "performance_headroom": max(0, (recommendation['estimated_throughput'] - target_qps) / target_qps * 100),
            "risk_level": self._assess_multi_cloud_risk_level(recommendation, target_qps, target_latency_ms),
            "cloud_provider_strengths": self._get_cloud_provider_strengths(recommendation['cloud_provider'])
        }
    
    def _assess_multi_cloud_risk_level(self, instance: Dict, target_qps: int, target_latency_ms: int) -> str:
        """Assess risk level based on performance margins and cloud provider."""
        
        qps_margin = (instance['estimated_throughput'] - target_qps) / target_qps
        latency_margin = (target_latency_ms - instance['estimated_latency']) / target_latency_ms
        
        # Adjust risk based on cloud provider reliability
        reliability_factor = self.cloud_reliability_scores.get(instance['cloud_provider'], 0.80)
        
        if qps_margin < 0.2 or latency_margin < 0.2:
            return "HIGH"
        elif qps_margin < 0.5 or latency_margin < 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_cloud_provider_strengths(self, cloud_provider: str) -> List[str]:
        """Get strengths of a specific cloud provider."""
        strengths = {
            "AWS": [
                "Industry-leading reliability and uptime",
                "Extensive global infrastructure",
                "Comprehensive AI/ML services",
                "Strong enterprise support"
            ],
            "Azure": [
                "Excellent enterprise integration",
                "Strong hybrid cloud capabilities",
                "Comprehensive compliance offerings",
                "Good cost optimization tools"
            ],
            "GCP": [
                "Often most cost-competitive",
                "Excellent AI/ML capabilities",
                "Strong data analytics",
                "Good performance for AI workloads"
            ],
            "Oracle": [
                "Aggressive pricing strategy",
                "Strong database integration",
                "Good for Oracle workloads",
                "Improving AI capabilities"
            ]
        }
        return strengths.get(cloud_provider, [])
    
    def _check_multi_cloud_sla_compliance(self, instance: Dict, sla_requirement: str) -> Dict:
        """Check SLA compliance for the recommended instance across cloud providers."""
        
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
            "cloud_provider_sla": self._get_cloud_provider_sla_info(instance['cloud_provider']),
            "recommended_actions": self._get_multi_cloud_sla_actions(instance, sla_requirement)
        }
    
    def _get_cloud_provider_sla_info(self, cloud_provider: str) -> Dict:
        """Get SLA information for a specific cloud provider."""
        sla_info = {
            "AWS": {
                "uptime_guarantee": 0.9995,
                "support_response": "24/7",
                "compensation": "10-100% credit"
            },
            "Azure": {
                "uptime_guarantee": 0.9995,
                "support_response": "24/7",
                "compensation": "10-100% credit"
            },
            "GCP": {
                "uptime_guarantee": 0.9995,
                "support_response": "24/7",
                "compensation": "10-100% credit"
            },
            "Oracle": {
                "uptime_guarantee": 0.999,
                "support_response": "24/7",
                "compensation": "10-100% credit"
            }
        }
        return sla_info.get(cloud_provider, {})
    
    def _get_multi_cloud_sla_actions(self, instance: Dict, sla_requirement: str) -> List[str]:
        """Get recommended actions for SLA compliance across cloud providers."""
        actions = []
        
        if sla_requirement == "enterprise":
            if instance['cloud_provider'] not in ['AWS', 'Azure']:
                actions.append(f"Consider {instance['cloud_provider']} enterprise support for better SLA guarantees")
            
            if instance['cpu'] < 8:
                actions.append("Consider higher CPU instance for better performance guarantees")
        
        if instance['reliability_score'] < 0.95:
            actions.append("Implement redundancy and failover mechanisms")
        
        if instance['cloud_provider'] == "Oracle":
            actions.append("Consider multi-region deployment for Oracle Cloud reliability")
        
        return actions
    
    def _identify_cost_savings_opportunities(self, scored_candidates: pd.DataFrame, target_qps: int, target_latency_ms: int) -> Dict:
        """Identify cost savings opportunities across cloud providers."""
        
        opportunities = {}
        
        # Find cheapest option that meets requirements
        cheapest_viable = scored_candidates[
            (scored_candidates['estimated_throughput'] >= target_qps) &
            (scored_candidates['estimated_latency'] <= target_latency_ms)
        ]
        
        if not cheapest_viable.empty:
            cheapest = cheapest_viable.iloc[0]
            most_expensive = cheapest_viable.iloc[-1]
            
            savings_potential = ((most_expensive['monthly_cost'] - cheapest['monthly_cost']) / most_expensive['monthly_cost']) * 100
            
            opportunities["cheapest_option"] = {
                "cloud_provider": cheapest['cloud_provider'],
                "instance": cheapest['instance_name'],
                "monthly_cost": cheapest['monthly_cost'],
                "savings_potential": savings_potential
            }
        
        # Cost optimization strategies by cloud provider
        strategies = {}
        for provider in get_cloud_providers():
            provider_strategies = get_cost_optimization_strategies(provider)
            if provider_strategies:
                strategies[provider] = provider_strategies
        
        opportunities["optimization_strategies"] = strategies
        
        return opportunities

# Legacy function for backward compatibility
def recommend_multi_cloud_instance(model_size_gb, qps, latency_ms, require_gpu=False):
    """Legacy recommendation function for backward compatibility."""
    optimizer = MultiCloudOptimizer()
    result = optimizer.recommend_multi_cloud(model_size_gb, qps, latency_ms, require_gpu)
    
    if "error" in result:
        return None
    
    primary = result["primary_recommendation"]
    return {
        "cloud_provider": primary["cloud_provider"],
        "name": primary["instance_name"],
        "cpu": primary["cpu"],
        "memory_gb": primary["memory_gb"],
        "gpu": primary["gpu"],
        "price_per_hour": primary["monthly_cost"] / (24 * 30)
    } 