def recommend_instance(model_size_gb, qps, latency_ms, require_gpu=False):
    """
    Recommend a GCP instance based on model size, QPS, latency, and GPU requirement.
    For MVP, use simple rules. In production, use benchmarking data.
    """
    from .gcp_data import get_gcp_instances

    df = get_gcp_instances()

    # Filter for GPU if required
    if require_gpu:
        df = df[df['gpu'] > 0]
    else:
        df = df[df['gpu'] == 0]

    # Filter by memory (model size + buffer)
    df = df[df['memory_gb'] >= model_size_gb * 2]

    # For MVP, just pick the cheapest that matches
    if not df.empty:
        return df.sort_values('price_per_hour').iloc[0].to_dict()
    else:
        return None
