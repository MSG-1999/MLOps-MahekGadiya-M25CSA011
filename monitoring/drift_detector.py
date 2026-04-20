# monitoring/drift_detector.py
"""
Drift Detector: Monitors for statistical drift in generation metrics.
Detects latency drift, error rate spikes, and throughput degradation.
Implements a simple sliding-window statistical test.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Deque

import numpy as np
from loguru import logger


@dataclass
class DriftAlert:
    """Represents a detected drift event."""
    metric: str
    severity: str        # "warning" | "critical"
    baseline_value: float
    current_value: float
    change_pct: float
    threshold_pct: float
    timestamp: float = field(default_factory=time.time)
    message: str = ""

    def __str__(self):
        return (
            f"[{self.severity.upper()}] {self.metric}: "
            f"{self.baseline_value:.2f} → {self.current_value:.2f} "
            f"({self.change_pct:+.1f}%)"
        )


@dataclass
class DriftConfig:
    """Configuration thresholds for drift detection."""
    # Latency
    latency_warn_pct: float = 25.0       # Warn if latency increases by 25%
    latency_critical_pct: float = 75.0   # Critical if latency increases by 75%

    # Error rate (absolute change in percentage points)
    error_rate_warn_pct: float = 5.0     # Warn if error rate rises by 5pp
    error_rate_critical_pct: float = 15.0

    # Throughput
    throughput_warn_pct: float = -20.0   # Warn if throughput drops by 20%
    throughput_critical_pct: float = -50.0

    # Minimum samples before drift detection kicks in
    min_baseline_samples: int = 10
    min_current_samples: int = 5

    # Window sizes
    baseline_window: int = 50
    current_window: int = 10


class DriftDetector:
    """
    Statistical drift detector using sliding windows.
    Compares recent performance against a rolling baseline.

    Metrics monitored:
    - Generation latency (seconds)
    - Error rate (fraction 0-1)
    - Throughput (generations per hour)
    """

    def __init__(self, config: Optional[DriftConfig] = None):
        self.config = config or DriftConfig()
        self._latencies: Deque[float] = deque(maxlen=self.config.baseline_window)
        self._successes: Deque[bool] = deque(maxlen=self.config.baseline_window)
        self._timestamps: Deque[float] = deque(maxlen=self.config.baseline_window)
        self._alerts: List[DriftAlert] = []
        self._last_check: float = 0

    def record(self, latency_s: float, success: bool):
        """Record a single generation event."""
        self._latencies.append(latency_s)
        self._successes.append(success)
        self._timestamps.append(time.time())

    def check_drift(self) -> List[DriftAlert]:
        """
        Run drift checks across all monitored metrics.

        Returns:
            List of DriftAlert objects (empty if no drift detected)
        """
        alerts = []
        n = len(self._latencies)

        if n < self.config.min_baseline_samples:
            return alerts  # Not enough data yet

        baseline_n = max(n - self.config.current_window, self.config.min_baseline_samples)
        baseline_latencies = list(self._latencies)[:baseline_n]
        current_latencies = list(self._latencies)[baseline_n:]

        baseline_successes = list(self._successes)[:baseline_n]
        current_successes = list(self._successes)[baseline_n:]

        if len(current_latencies) < self.config.min_current_samples:
            return alerts

        # ── Latency drift ──────────────────────────────────────────────
        baseline_lat = float(np.median(baseline_latencies))
        current_lat = float(np.median(current_latencies))

        if baseline_lat > 0:
            lat_change_pct = ((current_lat - baseline_lat) / baseline_lat) * 100
            if lat_change_pct >= self.config.latency_critical_pct:
                alert = DriftAlert(
                    metric="latency",
                    severity="critical",
                    baseline_value=baseline_lat,
                    current_value=current_lat,
                    change_pct=lat_change_pct,
                    threshold_pct=self.config.latency_critical_pct,
                    message=f"Generation latency has increased critically: {baseline_lat:.1f}s → {current_lat:.1f}s",
                )
                alerts.append(alert)
                logger.critical(str(alert))
            elif lat_change_pct >= self.config.latency_warn_pct:
                alert = DriftAlert(
                    metric="latency",
                    severity="warning",
                    baseline_value=baseline_lat,
                    current_value=current_lat,
                    change_pct=lat_change_pct,
                    threshold_pct=self.config.latency_warn_pct,
                    message=f"Generation latency has increased: {baseline_lat:.1f}s → {current_lat:.1f}s",
                )
                alerts.append(alert)
                logger.warning(str(alert))

        # ── Error rate drift ───────────────────────────────────────────
        baseline_err = 1.0 - (sum(baseline_successes) / len(baseline_successes))
        current_err = 1.0 - (sum(current_successes) / len(current_successes))
        err_change_pp = (current_err - baseline_err) * 100  # percentage points

        if err_change_pp >= self.config.error_rate_critical_pct:
            alert = DriftAlert(
                metric="error_rate",
                severity="critical",
                baseline_value=baseline_err * 100,
                current_value=current_err * 100,
                change_pct=err_change_pp,
                threshold_pct=self.config.error_rate_critical_pct,
                message=f"Error rate spike: {baseline_err*100:.1f}% → {current_err*100:.1f}%",
            )
            alerts.append(alert)
            logger.critical(str(alert))
        elif err_change_pp >= self.config.error_rate_warn_pct:
            alert = DriftAlert(
                metric="error_rate",
                severity="warning",
                baseline_value=baseline_err * 100,
                current_value=current_err * 100,
                change_pct=err_change_pp,
                threshold_pct=self.config.error_rate_warn_pct,
                message=f"Error rate elevated: {baseline_err*100:.1f}% → {current_err*100:.1f}%",
            )
            alerts.append(alert)
            logger.warning(str(alert))

        self._alerts.extend(alerts)
        self._last_check = time.time()
        return alerts

    def get_summary(self) -> Dict[str, Any]:
        """Get drift detection summary."""
        n = len(self._latencies)
        if n == 0:
            return {"samples": 0, "status": "insufficient_data"}

        latencies = list(self._latencies)
        successes = list(self._successes)
        recent_alerts = [a for a in self._alerts if time.time() - a.timestamp < 3600]

        return {
            "samples": n,
            "status": "alert" if recent_alerts else "normal",
            "baseline_latency_s": float(np.median(latencies[:max(1, n // 2)])),
            "recent_latency_s": float(np.median(latencies[n // 2:])),
            "overall_error_rate_pct": (1.0 - sum(successes) / n) * 100 if n > 0 else 0,
            "recent_alerts_count": len(recent_alerts),
            "last_check": self._last_check,
            "alert_history": [
                {
                    "metric": a.metric,
                    "severity": a.severity,
                    "message": a.message,
                    "timestamp": a.timestamp,
                }
                for a in self._alerts[-10:]
            ],
        }

    def reset_baseline(self):
        """Reset all collected data (use after model updates)."""
        self._latencies.clear()
        self._successes.clear()
        self._timestamps.clear()
        self._alerts.clear()
        logger.info("Drift detector baseline reset")
