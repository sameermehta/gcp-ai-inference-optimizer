import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import altair as alt
from streamlit_option_menu import option_menu

from multi_cloud_optimizer import MultiCloudOptimizer
from cloud_data import (
    get_all_cloud_instances, get_cloud_providers, get_cloud_regions,
    get_cloud_provider_comparison, get_cost_optimization_strategies
)

# Page configuration
st.set_page_config(
    page_title="Multi-Cloud AI Inference Optimizer",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e, #2ca02c, #d62728);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .cloud-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize optimizer
@st.cache_resource
def get_optimizer():
    return MultiCloudOptimizer()

optimizer = get_optimizer()

# Sidebar navigation
with st.sidebar:
    st.markdown("## ☁️ Multi-Cloud Optimizer")
    
    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "Multi-Cloud Optimizer", "Cost Comparison", "Performance Analysis", "Cost Optimization"],
        icons=["house", "cloud", "dollar", "graph-up", "trending-up"],
        menu_icon="cast",
        default_index=0,
    )

# Main dashboard
if selected == "Dashboard":
    st.markdown('<h1 class="main-header">Multi-Cloud AI Inference Optimizer</h1>', unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>💰 Cost Savings</h3>
            <h2>35.2%</h2>
            <p>vs. single cloud</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>☁️ Cloud Providers</h3>
            <h2>4</h2>
            <p>AWS, Azure, GCP, Oracle</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Performance</h3>
            <h2>2.8x</h2>
            <p>best-in-class</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>🛡️ SLA Compliance</h3>
            <h2>99.9%</h2>
            <p>enterprise grade</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Cloud provider overview
    st.subheader("☁️ Cloud Provider Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="cloud-card">
            <h3>🟠 AWS</h3>
            <p><strong>Market Leader</strong></p>
            <p>Best reliability & features</p>
            <p>Premium pricing</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="cloud-card">
            <h3>🔵 Azure</h3>
            <p><strong>Enterprise Focus</strong></p>
            <p>Great integration</p>
            <p>Competitive pricing</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="cloud-card">
            <h3>🟢 GCP</h3>
            <p><strong>Cost Leader</strong></p>
            <p>Best AI/ML capabilities</p>
            <p>Most competitive pricing</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="cloud-card">
            <h3>🔴 Oracle</h3>
            <p><strong>Emerging Player</strong></p>
            <p>Aggressive pricing</p>
            <p>Improving capabilities</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Cost Comparison by Cloud")
        
        # Sample cost comparison data
        clouds = ['AWS', 'Azure', 'GCP', 'Oracle']
        avg_costs = [0.25, 0.22, 0.18, 0.20]  # Average hourly costs
        
        fig = go.Figure(data=[go.Bar(x=clouds, y=avg_costs,
                                    marker_color=['#FF9900', '#0078D4', '#4285F4', '#F80000'])])
        fig.update_layout(
            title="Average Hourly Cost by Cloud Provider",
            xaxis_title="Cloud Provider",
            yaxis_title="Cost per Hour ($)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Performance vs Cost")
        
        # Performance vs cost scatter plot
        performance_data = {
            'Cloud': ['AWS', 'Azure', 'GCP', 'Oracle', 'AWS', 'Azure', 'GCP', 'Oracle'],
            'Instance': ['m5.xlarge', 'D4s_v3', 'e2-standard-8', 'Standard2.4', 'p3.2xlarge', 'NC6s_v3', 'a2-standard-8', 'BM.GPU3.8'],
            'Performance': [200, 180, 240, 100, 800, 750, 1200, 2000],
            'Cost': [0.192, 0.192, 0.268, 0.2, 3.06, 3.06, 1.412, 15.0]
        }
        
        df_perf = pd.DataFrame(performance_data)
        
        fig = px.scatter(df_perf, x='Cost', y='Performance', color='Cloud',
                        size='Performance', hover_data=['Instance'],
                        title="Performance vs Cost Analysis")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

elif selected == "Multi-Cloud Optimizer":
    st.markdown('<h1 class="main-header">Multi-Cloud Optimization Engine</h1>', unsafe_allow_html=True)
    
    # Input form
    with st.form("multi_cloud_optimization_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Workload Requirements")
            model_size_gb = st.number_input("Model Size (GB)", min_value=0.1, value=2.0, step=0.1)
            target_qps = st.number_input("Target QPS", min_value=1, value=100)
            target_latency_ms = st.number_input("Max Latency (ms)", min_value=1, value=50)
            require_gpu = st.checkbox("Require GPU", value=False)
        
        with col2:
            st.subheader("☁️ Cloud Preferences")
            preferred_clouds = st.multiselect(
                "Preferred Cloud Providers",
                get_cloud_providers(),
                default=get_cloud_providers()
            )
            sla_requirement = st.selectbox("SLA Level", ["standard", "premium", "enterprise"], index=0)
            budget_constraint = st.number_input("Max Hourly Budget ($)", min_value=0.0, value=5.0, step=0.1)
            
        submitted = st.form_submit_button("🚀 Get Multi-Cloud Recommendations", use_container_width=True)
    
    if submitted:
        with st.spinner("Analyzing optimal configurations across all cloud providers..."):
            result = optimizer.recommend_multi_cloud(
                model_size_gb=model_size_gb,
                target_qps=target_qps,
                target_latency_ms=target_latency_ms,
                require_gpu=require_gpu,
                budget_constraint=budget_constraint,
                preferred_clouds=preferred_clouds,
                sla_requirement=sla_requirement
            )
        
        if "error" in result:
            st.error(result["error"])
        else:
            primary = result["primary_recommendation"]
            
            # Primary recommendation
            st.success("✅ Optimal Multi-Cloud Configuration Found!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🏆 Best Overall Recommendation")
                st.metric("Cloud Provider", primary["cloud_provider"])
                st.metric("Instance", primary["instance_name"])
                st.metric("Monthly Cost", f"${primary['monthly_cost']:,.2f}")
                st.metric("Performance Score", f"{primary['total_score']:.2f}")
                
                # Performance analysis
                perf_analysis = result["performance_analysis"]
                st.markdown("### 📊 Performance Analysis")
                st.metric("QPS Capacity", f"{perf_analysis['qps_capacity']:,}")
                st.metric("Utilization", f"{perf_analysis['qps_utilization']:.1f}%")
                st.metric("Latency", f"{perf_analysis['latency_performance']}ms")
                
                # Risk assessment
                risk_color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}
                st.markdown(f"### ⚠️ Risk Level: :{risk_color[perf_analysis['risk_level']]}[{perf_analysis['risk_level']}]")
            
            with col2:
                # Instance specifications
                st.markdown("### ⚙️ Specifications")
                specs_data = {
                    "CPU": f"{primary['cpu']} vCPUs",
                    "Memory": f"{primary['memory_gb']} GB",
                    "GPU": f"{primary['gpu']} GPU(s)" if primary['gpu'] > 0 else "None",
                    "Network": primary['network_performance'],
                    "Region": primary['region']
                }
                
                for key, value in specs_data.items():
                    st.metric(key, value)
                
                # Cloud provider strengths
                st.markdown("### 🌟 Cloud Provider Strengths")
                strengths = perf_analysis['cloud_provider_strengths']
                for strength in strengths[:3]:  # Show top 3
                    st.markdown(f"• {strength}")
            
            # Recommendations by cloud provider
            st.markdown("### ☁️ Recommendations by Cloud Provider")
            
            recommendations_by_provider = result["recommendations_by_provider"]
            
            for provider, instances in recommendations_by_provider.items():
                st.markdown(f"#### {provider}")
                
                provider_df = pd.DataFrame(instances)
                display_df = provider_df[["instance_name", "monthly_cost", "total_score", "estimated_latency", "estimated_throughput"]]
                display_df.columns = ["Instance", "Monthly Cost", "Score", "Latency (ms)", "Throughput (QPS)"]
                display_df["Monthly Cost"] = display_df["Monthly Cost"].apply(lambda x: f"${x:,.2f}")
                display_df["Score"] = display_df["Score"].apply(lambda x: f"{x:.2f}")
                
                st.dataframe(display_df, use_container_width=True)

elif selected == "Cost Comparison":
    st.markdown('<h1 class="main-header">Multi-Cloud Cost Analysis</h1>', unsafe_allow_html=True)
    
    # Cloud provider comparison table
    st.subheader("📊 Cloud Provider Cost Comparison")
    
    comparison_df = get_cloud_provider_comparison()
    st.dataframe(comparison_df, use_container_width=True)
    
    # Cost comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Cost Range by Provider")
        
        # Cost range chart
        fig = go.Figure()
        
        for _, row in comparison_df.iterrows():
            provider = row['Cloud Provider']
            min_cost = row['Min Hourly Cost']
            max_cost = row['Max Hourly Cost']
            avg_cost = row['Avg Hourly Cost']
            
            fig.add_trace(go.Bar(
                name=provider,
                x=[provider],
                y=[max_cost],
                base=[min_cost],
                marker_color=['#FF9900', '#0078D4', '#4285F4', '#F80000'][get_cloud_providers().index(provider)]
            ))
        
        fig.update_layout(
            title="Cost Range by Cloud Provider",
            xaxis_title="Cloud Provider",
            yaxis_title="Hourly Cost ($)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Instance Count by Provider")
        
        # Instance count chart
        fig = go.Figure(data=[go.Pie(
            labels=comparison_df['Cloud Provider'],
            values=comparison_df['Instance Count'],
            hole=0.3
        )])
        fig.update_layout(title="Instance Types Available", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed cost analysis
    st.subheader("🔍 Detailed Cost Analysis")
    
    # Get all instances for detailed analysis
    all_instances = get_all_cloud_instances()
    
    # Filter by instance type (CPU-based)
    cpu_ranges = [(1, 2), (3, 4), (5, 8), (9, 16), (17, 32)]
    
    for cpu_min, cpu_max in cpu_ranges:
        filtered_instances = all_instances[
            (all_instances['cpu'] >= cpu_min) & (all_instances['cpu'] <= cpu_max)
        ]
        
        if not filtered_instances.empty:
            st.markdown(f"#### {cpu_min}-{cpu_max} vCPU Instances")
            
            # Create comparison chart
            fig = px.box(filtered_instances, x='cloud_provider', y='price_per_hour',
                        title=f"Cost Distribution for {cpu_min}-{cpu_max} vCPU Instances")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

elif selected == "Performance Analysis":
    st.markdown('<h1 class="main-header">Multi-Cloud Performance Analysis</h1>', unsafe_allow_html=True)
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Best Performance", "GCP A2", "2.4x faster")
    with col2:
        st.metric("Best Value", "GCP E2", "35% cheaper")
    with col3:
        st.metric("Most Reliable", "AWS", "99.99% uptime")
    with col4:
        st.metric("Best GPU", "AWS P3", "V100 GPUs")
    
    # Performance comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚡ Performance vs Cost Efficiency")
        
        # Sample performance data
        performance_data = {
            'Cloud': ['AWS', 'Azure', 'GCP', 'Oracle'],
            'Performance_Score': [85, 82, 95, 75],
            'Cost_Efficiency': [80, 85, 95, 90],
            'Reliability': [99, 98, 97, 95]
        }
        
        df_perf = pd.DataFrame(performance_data)
        
        fig = px.scatter(df_perf, x='Cost_Efficiency', y='Performance_Score', 
                        size='Reliability', color='Cloud',
                        title="Performance vs Cost Efficiency")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Latency Distribution")
        
        # Simulate latency data for different clouds
        np.random.seed(42)
        latency_data = []
        
        for cloud in get_cloud_providers():
            # Different latency characteristics per cloud
            if cloud == "AWS":
                latencies = np.random.normal(25, 8, 100)
            elif cloud == "Azure":
                latencies = np.random.normal(30, 10, 100)
            elif cloud == "GCP":
                latencies = np.random.normal(20, 6, 100)
            else:  # Oracle
                latencies = np.random.normal(35, 12, 100)
            
            for latency in latencies:
                latency_data.append({'Cloud': cloud, 'Latency (ms)': max(5, latency)})
        
        df_latency = pd.DataFrame(latency_data)
        
        fig = px.histogram(df_latency, x='Latency (ms)', color='Cloud',
                          title="Latency Distribution by Cloud Provider",
                          nbins=20)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance heatmap
    st.subheader("🔥 Performance Heatmap")
    
    # Create performance matrix
    performance_matrix = {
        'Metric': ['Cost Efficiency', 'Performance', 'Reliability', 'Scalability', 'GPU Support'],
        'AWS': [80, 85, 99, 95, 95],
        'Azure': [85, 82, 98, 90, 85],
        'GCP': [95, 95, 97, 92, 90],
        'Oracle': [90, 75, 95, 80, 70]
    }
    
    df_matrix = pd.DataFrame(performance_matrix)
    df_matrix_melted = df_matrix.melt(id_vars=['Metric'], var_name='Cloud', value_name='Score')
    
    fig = px.imshow(
        df_matrix.set_index('Metric'),
        title="Performance Matrix by Cloud Provider",
        color_continuous_scale='RdYlGn',
        aspect='auto'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

elif selected == "Cost Optimization":
    st.markdown('<h1 class="main-header">Multi-Cloud Cost Optimization</h1>', unsafe_allow_html=True)
    
    # Cost savings overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="success-box">
            <h3>💰 Reserved Instances</h3>
            <p>Up to 60% savings</p>
            <p><strong>Best: AWS</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
            <h3>⚡ Spot Instances</h3>
            <p>Up to 90% savings</p>
            <p><strong>Best: AWS</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-box">
            <h3>📈 Auto Scaling</h3>
            <p>Up to 45% savings</p>
            <p><strong>Best: GCP</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="success-box">
            <h3>🎯 Right Sizing</h3>
            <p>Up to 25% savings</p>
            <p><strong>All Clouds</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Cost optimization strategies by cloud
    st.subheader("📊 Cost Optimization Strategies by Cloud Provider")
    
    for provider in get_cloud_providers():
        st.markdown(f"#### {provider}")
        
        strategies = get_cost_optimization_strategies(provider)
        
        if strategies:
            strategy_df = pd.DataFrame([
                {
                    'Strategy': strategy,
                    'Savings': f"{data['savings']*100:.0f}%",
                    'Commitment': data.get('commitment', data.get('availability', data.get('effort', 'N/A'))),
                    'Description': data['description']
                }
                for strategy, data in strategies.items()
            ])
            
            st.dataframe(strategy_df, use_container_width=True)
        else:
            st.info(f"No specific optimization strategies available for {provider}")
    
    # Cost optimization recommendations
    st.subheader("🎯 Personalized Cost Optimization Recommendations")
    
    # Interactive form for recommendations
    with st.form("cost_optimization_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            current_monthly_spend = st.number_input("Current Monthly Spend ($)", min_value=100, value=1000)
            current_cloud = st.selectbox("Current Cloud Provider", get_cloud_providers())
            workload_type = st.selectbox("Workload Type", ["Production", "Development", "Testing", "Batch Processing"])
        
        with col2:
            optimization_goals = st.multiselect(
                "Optimization Goals",
                ["Reduce Costs", "Improve Performance", "Increase Reliability", "Simplify Management"]
            )
            risk_tolerance = st.selectbox("Risk Tolerance", ["Low", "Medium", "High"])
        
        optimize_submitted = st.form_submit_button("Get Optimization Recommendations")
    
    if optimize_submitted:
        st.success("### 💡 Cost Optimization Recommendations")
        
        recommendations = [
            f"**Switch to {current_cloud} Reserved Instances**: Save up to 60% on predictable workloads",
            f"**Consider {current_cloud} Spot Instances**: Save up to 90% for flexible workloads",
            f"**Implement Auto Scaling**: Reduce idle costs by 40-50%",
            f"**Right-size instances**: Optimize CPU/memory allocation for 25% savings",
            f"**Multi-region optimization**: Choose cost-effective regions",
            f"**Storage optimization**: Use appropriate storage tiers"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
        
        # Estimated savings
        estimated_savings = current_monthly_spend * 0.35  # 35% average savings
        st.metric("Estimated Monthly Savings", f"${estimated_savings:,.2f}")
        st.metric("Annual Savings", f"${estimated_savings * 12:,.2f}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Multi-Cloud AI Inference Optimizer v3.0 | Enterprise Multi-Cloud Solution</p>
</div>
""", unsafe_allow_html=True) 