from prometheus_client import CollectorRegistry, Gauge

metrics_registry = CollectorRegistry()
request_counter = Gauge("easygo_ai_requests_total", "Total AI API requests", registry=metrics_registry)
