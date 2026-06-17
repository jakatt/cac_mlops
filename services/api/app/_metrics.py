"""Prometheus metrics registry shared across the API."""
from prometheus_client import CollectorRegistry, Counter

REGISTRY = CollectorRegistry(auto_describe=True)

REQUESTS_TOTAL = Counter(
    "api_requests_total",
    "Total HTTP requests",
    ["endpoint", "method", "status"],
    registry=REGISTRY,
)

PREDICTIONS_TOTAL = Counter(
    "api_predictions_total",
    "Total predictions by result class",
    ["result"],
    registry=REGISTRY,
)
