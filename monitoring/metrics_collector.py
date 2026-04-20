# monitoring/metrics_collector.py
"""
Metrics Collector: Real-time performance tracking with Prometheus and in-memory stats.
Tracks latency, throughput, error rates, and hardware utilization.
"""

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Deque

import numpy as np
from loguru import logger

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, start_http_server, CollectorRegistry
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus-client not installed. Prometheus metrics disabled.")


@dataclass
class GenerationMetric:
    """Single generation event metrics."""
    request_id: str
    timestamp: float
    generation_time_s: float
    steps: int
    width: int
    height: int
    num_images: int
    scheduler: str
    success: bool
    error: Optional[str] = None

    @property
    def steps_per_second(self) -> float:
        if self.generation_time_s <= 0:
            return 0
        return self.steps / self.generation_time_s

    @property
    def pixels_per_second(self) -> float:
        if self.generation_time_s <= 0:
            return 0
        return (self.width * self.height * self.num_images) / self.generation_time_s


class MetricsCollector:
    """
    Collects and aggregates generation metrics.
    Provides Prometheus export and in-memory rolling stats.
    """

    def __init__(
        self,
        window_size: int = 100,
        prometheus_port: int = 8012,
        enable_prometheus: bool = True,
    ):
        self.window_size = window_size
        self._lock = threading.Lock()
        self._metrics: Deque[GenerationMetric] = deque(maxlen=window_size)
        self._start_time = time.time()
        self._total_generations = 0
        self._total_errors = 0

        # Prometheus metrics
        self._prometheus_enabled = False
        if enable_prometheus and PROMETHEUS_AVAILABLE:
            self._setup_prometheus(prometheus_port)

    def _setup_prometheus(self, port: int):
        """Initialize Prometheus metrics and HTTP server."""
        try:
            self.prom_generation_total = Counter(
                "sd_generations_total",
                "Total number of generation requests",
                ["status", "scheduler"]
            )
            self.prom_generation_duration = Histogram(
                "sd_generation_duration_seconds",
                "Generation duration in seconds",
                buckets=[5, 10, 20, 30, 45, 60, 90, 120, 180, 300]
            )
            self.prom_steps_per_second = Gauge(
                "sd_steps_per_second",
                "Current inference speed in steps/sec"
            )
            self.prom_active_generations = Gauge(
                "sd_active_generations",
                "Number of currently running generations"
            )
            self.prom_gpu_memory_used = Gauge(
                "sd_gpu_memory_used_bytes",
                "GPU memory currently allocated"
            )

            start_http_server(port)
            self._prometheus_enabled = True
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            if "Address already in use" in str(e):
                logger.error(
                    f"Prometheus port {port} is already in use. "
                    "This usually happens if an orphaned Streamlit process is still running. "
                    "Use 'kill $(pgrep -f streamlit)' or check port status before restarting."
                )
            else:
                logger.warning(f"Prometheus setup failed: {e}")

    def record_generation(self, metric: GenerationMetric):
        """Record a completed generation."""
        with self._lock:
            self._metrics.append(metric)
            self._total_generations += 1
            if not metric.success:
                self._total_errors += 1

        if self._prometheus_enabled:
            try:
                status = "success" if metric.success else "error"
                self.prom_generation_total.labels(
                    status=status,
                    scheduler=metric.scheduler
                ).inc()
                self.prom_generation_duration.observe(metric.generation_time_s)
                self.prom_steps_per_second.set(metric.steps_per_second)
            except Exception as e:
                logger.debug(f"Prometheus record failed: {e}")

        # Update GPU metrics
        self._update_gpu_metrics()

    def _update_gpu_metrics(self):
        """Update GPU memory Prometheus gauges."""
        if not (TORCH_AVAILABLE and self._prometheus_enabled):
            return
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                self.prom_gpu_memory_used.set(allocated)
        except Exception:
            pass

    def set_active_generations(self, count: int):
        """Update active generation counter."""
        if self._prometheus_enabled:
            try:
                self.prom_active_generations.set(count)
            except Exception:
                pass

    def get_rolling_stats(self) -> Dict[str, Any]:
        """Compute rolling statistics over the recent window."""
        with self._lock:
            metrics = list(self._metrics)

        if not metrics:
            return self._empty_stats()

        successful = [m for m in metrics if m.success]
        failed = [m for m in metrics if not m.success]

        latencies = [m.generation_time_s for m in successful]
        speeds = [m.steps_per_second for m in successful]

        uptime = time.time() - self._start_time

        return {
            "window_size": len(metrics),
            "total_generations": self._total_generations,
            "total_errors": self._total_errors,
            "error_rate": len(failed) / len(metrics) if metrics else 0,
            "success_rate": len(successful) / len(metrics) if metrics else 1.0,
            "uptime_hours": uptime / 3600,
            "throughput_per_hour": self._total_generations / (uptime / 3600) if uptime > 0 else 0,
            "latency": {
                "mean_s": float(np.mean(latencies)) if latencies else 0,
                "median_s": float(np.median(latencies)) if latencies else 0,
                "p95_s": float(np.percentile(latencies, 95)) if latencies else 0,
                "p99_s": float(np.percentile(latencies, 99)) if latencies else 0,
                "min_s": float(np.min(latencies)) if latencies else 0,
                "max_s": float(np.max(latencies)) if latencies else 0,
            },
            "speed": {
                "mean_steps_per_s": float(np.mean(speeds)) if speeds else 0,
                "max_steps_per_s": float(np.max(speeds)) if speeds else 0,
            },
            "scheduler_usage": self._scheduler_counts(metrics),
        }

    def _scheduler_counts(self, metrics: List[GenerationMetric]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for m in metrics:
            counts[m.scheduler] = counts.get(m.scheduler, 0) + 1
        return counts

    def _empty_stats(self) -> Dict[str, Any]:
        return {
            "window_size": 0,
            "total_generations": self._total_generations,
            "total_errors": self._total_errors,
            "error_rate": 0,
            "success_rate": 1.0,
            "uptime_hours": (time.time() - self._start_time) / 3600,
            "throughput_per_hour": 0,
            "latency": {"mean_s": 0, "median_s": 0, "p95_s": 0, "p99_s": 0, "min_s": 0, "max_s": 0},
            "speed": {"mean_steps_per_s": 0, "max_steps_per_s": 0},
            "scheduler_usage": {},
        }

    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get current GPU hardware stats."""
        if not TORCH_AVAILABLE:
            return {"available": False}

        if not torch.cuda.is_available():
            return {"available": False, "device": "cpu"}

        return {
            "available": True,
            "device_name": torch.cuda.get_device_name(0),
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            "total_mb": torch.cuda.get_device_properties(0).total_memory / 1024**2,
            "utilization_pct": (
                torch.cuda.memory_allocated() /
                torch.cuda.get_device_properties(0).total_memory * 100
            ),
        }

    def get_recent_history(self, n: int = 20) -> List[Dict[str, Any]]:
        """Get recent N generation metrics as dicts."""
        with self._lock:
            recent = list(self._metrics)[-n:]
        return [
            {
                "request_id": m.request_id,
                "timestamp": m.timestamp,
                "generation_time_s": round(m.generation_time_s, 2),
                "steps_per_second": round(m.steps_per_second, 2),
                "steps": m.steps,
                "width": m.width,
                "height": m.height,
                "num_images": m.num_images,
                "scheduler": m.scheduler,
                "success": m.success,
                "error": m.error,
            }
            for m in reversed(recent)
        ]
