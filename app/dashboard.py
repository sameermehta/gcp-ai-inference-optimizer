import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import altair as alt
from streamlit_option_menu import option_menu
from streamlit_aggrid import AgGrid, GridOptionsBuilder
from streamlit_plotly_events import plotly_events

from app.optimizer import EnterpriseOptimizer
from app.gcp_data import get_gcp_instances, get_regions, get_historical_costs, get_instance_utilization

# Page configuration
st.set_page_config(
    page_title="Enterprise AI Inference Optimizer",
    page_icon="🚀",
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
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
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
    return EnterpriseOptimizer()

optimizer = get_optimizer()

# Sidebar navigation
with st.sidebar:
    st.markdown("## 🚀 Enterprise AI Optimizer")
    
    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "Instance Optimizer", "Cost Analysis", "Performance Monitoring", "SLA Compliance"],
        icons=["house", "gear", "dollar", "graph-up", "shield-check"],
        menu_icon="cast",
        default_index=0,
    )

# Main dashboard
if selected == "Dashboard":
    st.markdown('<h1 class="main-header">Enterprise AI Inference Optimizer</h1>', unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>💰 Cost Savings</h3>
            <h2>23.5%</h2>
            <p>vs. current setup</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Performance</h3>
            <h2>2.4x</h2>
            <p>throughput improvement</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🛡️ SLA Compliance</h3>
            <h2>99.9%</h2>
            <p>uptime guarantee</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Instances</h3>
            <h2>156</h2>
            <p>optimized this month</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Cost Trends")
        # Simulate cost trend data
        dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='M')
        costs = [12000, 11800, 11500, 11200, 11000, 10800, 10500, 10200, 10000, 9800, 9600, 9400]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=costs, mode='lines+markers', 
                                line=dict(color='#1f77b4', width=3),
                                marker=dict(size=8)))
        fig.update_layout(
            title="Monthly Infrastructure Costs",
            xaxis_title="Month",
            yaxis_title="Cost ($)",
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Performance Distribution")
        # Performance distribution chart
        instance_types = ['E2', 'C2', 'N1', 'A2']
        performance_scores = [85, 92, 88, 95]
        
        fig = go.Figure(data=[go.Bar(x=instance_types, y=performance_scores,
                                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])])
        fig.update_layout(
            title="Performance Scores by Instance Family",
            xaxis_title="Instance Family",
            yaxis_title="Performance Score",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

elif selected == "Instance Optimizer":
    st.markdown('<h1 class="main-header">Instance Optimization Engine</h1>', unsafe_allow_html=True)
    
    # Input form
    with st.form("optimization_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Workload Requirements")
            model_size_gb = st.number_input("Model Size (GB)", min_value=0.1, value=2.0, step=0.1)
            target_qps = st.number_input("Target QPS", min_value=1, value=100)
            target_latency_ms = st.number_input("Max Latency (ms)", min_value=1, value=50)
            require_gpu = st.checkbox("Require GPU", value=False)
        
        with col2:
            st.subheader("🏢 Enterprise Settings")
            region = st.selectbox("GCP Region", get_regions(), index=0)
            sla_requirement = st.selectbox("SLA Level", ["standard", "premium", "enterprise"], index=0)
            budget_constraint = st.number_input("Max Hourly Budget ($)", min_value=0.0, value=5.0, step=0.1)
            
        submitted = st.form_submit_button("🚀 Get Recommendations", use_container_width=True)
    
    if submitted:
        with st.spinner("Analyzing optimal configurations..."):
            result = optimizer.recommend_instance(
                model_size_gb=model_size_gb,
                target_qps=target_qps,
                target_latency_ms=target_latency_ms,
                require_gpu=require_gpu,
                budget_constraint=budget_constraint,
                region=region,
                sla_requirement=sla_requirement
            )
        
        if "error" in result:
            st.error(result["error"])
        else:
            primary = result["primary_recommendation"]
            
            # Primary recommendation
            st.success("✅ Optimal Configuration Found!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🏆 Primary Recommendation")
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
                    "Family": primary['family'].upper()
                }
                
                for key, value in specs_data.items():
                    st.metric(key, value)
                
                # SLA compliance
                sla_info = result["sla_compliance"]
                st.markdown("### 🛡️ SLA Compliance")
                st.metric("SLA Level", sla_info["sla_level"].title())
                st.metric("Uptime", f"{sla_info['uptime_requirement']:.1%}")
                st.metric("Compliance Score", f"{sla_info['compliance_score']:.2f}")
            
            # Alternative recommendations
            if result["alternative_recommendations"]:
                st.markdown("### 🔄 Alternative Options")
                
                alt_df = pd.DataFrame(result["alternative_recommendations"])
                alt_df = alt_df[["instance_name", "monthly_cost", "total_score", "estimated_latency", "estimated_throughput"]]
                alt_df.columns = ["Instance", "Monthly Cost", "Score", "Latency (ms)", "Throughput (QPS)"]
                alt_df["Monthly Cost"] = alt_df["Monthly Cost"].apply(lambda x: f"${x:,.2f}")
                alt_df["Score"] = alt_df["Score"].apply(lambda x: f"{x:.2f}")
                
                st.dataframe(alt_df, use_container_width=True)

elif selected == "Cost Analysis":
    st.markdown('<h1 class="main-header">Cost Optimization Center</h1>', unsafe_allow_html=True)
    
    # Cost comparison chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Cost vs Performance")
        
        # Sample cost-performance data
        instances = ["E2-Standard-4", "C2-Standard-8", "N1-Standard-8", "A2-Standard-8"]
        costs = [0.134, 0.416, 0.380, 1.412]
        performance = [120, 280, 240, 1200]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=costs,
            y=performance,
            mode='markers+text',
            text=instances,
            textposition="top center",
            marker=dict(size=15, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ))
        fig.update_layout(
            title="Cost vs Performance Analysis",
            xaxis_title="Hourly Cost ($)",
            yaxis_title="Performance (QPS)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Cost Breakdown")
        
        # Cost breakdown pie chart
        categories = ["Compute", "GPU", "Network", "Storage", "Support"]
        costs = [45, 35, 10, 8, 2]
        
        fig = go.Figure(data=[go.Pie(labels=categories, values=costs, hole=0.3)])
        fig.update_layout(title="Monthly Cost Breakdown", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Historical cost analysis
    st.subheader("📈 Historical Cost Trends")
    
    # Simulate historical data for multiple instances
    dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='D')
    
    # Create sample historical data
    np.random.seed(42)
    base_costs = {
        "E2-Standard-4": 0.134,
        "C2-Standard-8": 0.416,
        "A2-Standard-8": 1.412
    }
    
    historical_data = []
    for instance, base_cost in base_costs.items():
        for date in dates:
            # Add some variation
            variation = 1 + np.random.normal(0, 0.05)
            daily_cost = base_cost * 24 * variation
            historical_data.append({
                "Date": date,
                "Instance": instance,
                "Daily Cost": daily_cost
            })
    
    df_historical = pd.DataFrame(historical_data)
    
    fig = px.line(df_historical, x="Date", y="Daily Cost", color="Instance",
                  title="Daily Cost Trends by Instance Type")
    st.plotly_chart(fig, use_container_width=True)

elif selected == "Performance Monitoring":
    st.markdown('<h1 class="main-header">Performance Monitoring Dashboard</h1>', unsafe_allow_html=True)
    
    # Real-time metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("CPU Utilization", "78.5%", "2.3%")
    with col2:
        st.metric("Memory Usage", "65.2%", "-1.1%")
    with col3:
        st.metric("GPU Utilization", "89.3%", "5.7%")
    with col4:
        st.metric("Network I/O", "2.4 Gbps", "0.3 Gbps")
    
    # Performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚡ Latency Distribution")
        
        # Simulate latency data
        latencies = np.random.normal(25, 5, 1000)
        fig = px.histogram(x=latencies, nbins=30, title="Response Time Distribution")
        fig.update_layout(xaxis_title="Latency (ms)", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Throughput Over Time")
        
        # Simulate throughput data
        time_points = pd.date_range(start='2024-01-01', periods=100, freq='H')
        throughput = np.random.normal(800, 50, 100) + np.sin(np.arange(100) * 0.1) * 100
        
        fig = px.line(x=time_points, y=throughput, title="QPS Over Time")
        fig.update_layout(xaxis_title="Time", yaxis_title="Queries Per Second")
        st.plotly_chart(fig, use_container_width=True)
    
    # Instance utilization heatmap
    st.subheader("🔥 Instance Utilization Heatmap")
    
    # Create sample utilization data
    instances = ["E2-Standard-4", "C2-Standard-8", "A2-Standard-8", "N1-Standard-16"]
    metrics = ["CPU", "Memory", "GPU", "Network", "Disk"]
    
    utilization_data = np.random.uniform(0.3, 0.9, (len(instances), len(metrics)))
    
    fig = go.Figure(data=go.Heatmap(
        z=utilization_data,
        x=metrics,
        y=instances,
        colorscale='RdYlGn_r'
    ))
    fig.update_layout(title="Resource Utilization by Instance")
    st.plotly_chart(fig, use_container_width=True)

elif selected == "SLA Compliance":
    st.markdown('<h1 class="main-header">SLA Compliance Center</h1>', unsafe_allow_html=True)
    
    # SLA status overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="success-box">
            <h3>✅ Standard SLA</h3>
            <p>99.0% uptime requirement</p>
            <p><strong>Current: 99.8%</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
            <h3>⚠️ Premium SLA</h3>
            <p>99.5% uptime requirement</p>
            <p><strong>Current: 99.4%</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-box">
            <h3>ℹ️ Enterprise SLA</h3>
            <p>99.9% uptime requirement</p>
            <p><strong>Current: 99.7%</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # SLA compliance chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 SLA Compliance Trends")
        
        # Simulate SLA compliance data
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        standard_sla = [99.8, 99.9, 99.7, 99.8, 99.9, 99.8]
        premium_sla = [99.4, 99.6, 99.3, 99.5, 99.4, 99.4]
        enterprise_sla = [99.7, 99.8, 99.6, 99.7, 99.8, 99.7]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months, y=standard_sla, name="Standard", line=dict(color='green')))
        fig.add_trace(go.Scatter(x=months, y=premium_sla, name="Premium", line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=months, y=enterprise_sla, name="Enterprise", line=dict(color='red')))
        
        fig.update_layout(
            title="Monthly SLA Compliance",
            xaxis_title="Month",
            yaxis_title="Uptime (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 SLA Recommendations")
        
        recommendations = [
            "Upgrade to A2 instances for better performance guarantees",
            "Implement auto-scaling for traffic spikes",
            "Add redundant instances for critical workloads",
            "Monitor network latency and optimize routing",
            "Consider multi-region deployment for global users"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"**{i}.** {rec}")
    
    # SLA violation analysis
    st.subheader("🚨 SLA Violation Analysis")
    
    # Sample violation data
    violations_data = {
        "Date": ["2024-01-15", "2024-02-03", "2024-03-10", "2024-04-22"],
        "Instance": ["E2-Standard-4", "C2-Standard-8", "A2-Standard-8", "N1-Standard-16"],
        "Violation Type": ["High Latency", "Downtime", "Performance Degradation", "Network Issue"],
        "Duration": ["15 min", "8 min", "25 min", "12 min"],
        "Impact": ["Medium", "High", "Low", "Medium"]
    }
    
    violations_df = pd.DataFrame(violations_data)
    st.dataframe(violations_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Enterprise AI Inference Optimizer v2.0 | Built for Enterprise Scale</p>
</div>
""", unsafe_allow_html=True)
