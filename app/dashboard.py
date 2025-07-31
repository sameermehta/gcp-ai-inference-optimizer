import streamlit as st
from app.optimizer import recommend_instance

st.title("GCP AI Inference Optimizer")

st.sidebar.header("Workload Requirements")
model_size_gb = st.sidebar.number_input("Model Size (GB)", min_value=0.1, value=1.0, step=0.1)
qps = st.sidebar.number_input("Queries Per Second (QPS)", min_value=1, value=10)
latency_ms = st.sidebar.number_input("Max Latency (ms)", min_value=1, value=100)
require_gpu = st.sidebar.checkbox("Require GPU", value=False)

if st.sidebar.button("Get Recommendation"):
    rec = recommend_instance(model_size_gb, qps, latency_ms, require_gpu)
    if rec:
        st.success(f"Recommended Instance: {rec['name']}")
        st.write(f"Specs: {rec['cpu']} vCPU, {rec['memory_gb']} GB RAM, {rec['gpu']} GPU(s)")
        st.write(f"Estimated Price: ${rec['price_per_hour']}/hour")
    else:
        st.error("No suitable instance found for the given requirements.")
