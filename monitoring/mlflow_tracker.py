# monitoring/mlflow_tracker.py
"""
MLflow Tracker: Experiment tracking, metric logging, and artifact management
for the Stable Diffusion MLOps pipeline.
"""

import io
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

from PIL import Image
from loguru import logger

try:
    import mlflow
    import mlflow.artifacts
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed. Tracking disabled.")


class MLflowTracker:
    """
    MLflow-based experiment tracker for SD generation runs.
    Logs prompts, parameters, metrics, and output images.
    Gracefully degrades if MLflow is unavailable.
    """

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5012",
        experiment_name: str = "stable-diffusion-v1-5",
        artifact_location: Optional[str] = None,
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.artifact_location = artifact_location
        self.enabled = MLFLOW_AVAILABLE
        self._active_run = None
        self._experiment_id: Optional[str] = None

        if self.enabled:
            self._setup()

    def _setup(self):
        """Initialize MLflow connection and experiment."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                kwargs = {"name": self.experiment_name}
                if self.artifact_location:
                    kwargs["artifact_location"] = self.artifact_location
                self._experiment_id = mlflow.create_experiment(**kwargs)
                logger.info(f"Created MLflow experiment: {self.experiment_name}")
            else:
                self._experiment_id = experiment.experiment_id
                logger.info(f"Using MLflow experiment: {self.experiment_name} (id={self._experiment_id})")
        except Exception as e:
            logger.warning(f"MLflow setup failed (tracking disabled): {e}")
            self.enabled = False

    @contextmanager
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict] = None):
        """Context manager for a generation run."""
        if not self.enabled:
            yield None
            return

        try:
            with mlflow.start_run(
                experiment_id=self._experiment_id,
                run_name=run_name or f"generation_{int(time.time())}",
                tags=tags or {},
            ) as run:
                self._active_run = run
                yield run
        except Exception as e:
            logger.warning(f"MLflow run failed: {e}")
            yield None
        finally:
            self._active_run = None

    def log_generation(
        self,
        prompt: str,
        negative_prompt: str,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        images: List[Image.Image],
        seed: int,
        request_id: str,
    ):
        """
        Log a complete generation event to MLflow.

        Logs:
        - Parameters: all generation settings
        - Metrics: latency, quality indicators
        - Artifacts: generated images
        - Tags: prompt text, model info
        """
        if not self.enabled:
            return

        run_name = f"gen_{request_id}"
        tags = {
            "prompt": prompt[:250],
            "model": "stable-diffusion-v1-5",
            "scheduler": config.get("scheduler", "unknown"),
        }

        try:
            with mlflow.start_run(
                experiment_id=self._experiment_id,
                run_name=run_name,
                tags=tags,
            ):
                # Log parameters
                params = {
                    "prompt_length": len(prompt),
                    "negative_prompt_length": len(negative_prompt),
                    "width": config.get("width", 512),
                    "height": config.get("height", 512),
                    "steps": config.get("num_inference_steps", 20),
                    "guidance_scale": config.get("guidance_scale", 7.5),
                    "seed": seed,
                    "num_images": config.get("num_images", 1),
                    "scheduler": config.get("scheduler", "DDIM"),
                    "style_preset": config.get("style_preset", "None"),
                    "quality_boost": config.get("quality_boost", False),
                }
                mlflow.log_params(params)

                # Log metrics
                mlflow.log_metrics({
                    "generation_time_s": metrics.get("generation_time_s", 0),
                    "steps_per_second": metrics.get("steps_per_second", 0),
                    "images_generated": len(images),
                    "pixels_generated": config.get("width", 512) * config.get("height", 512) * len(images),
                })

                # Log full prompts as text artifacts
                with open("/tmp/prompt.txt", "w") as f:
                    f.write(f"PROMPT:\n{prompt}\n\nNEGATIVE:\n{negative_prompt}")
                mlflow.log_artifact("/tmp/prompt.txt", "prompts")

                # Log images
                for i, img in enumerate(images[:4]):  # cap at 4 to save storage
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    buf.seek(0)
                    mlflow.log_image(img, f"generated_image_{i}.png")

        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")

    def log_model_load(self, model_info: Dict[str, Any], load_time_s: float):
        """Log model loading event."""
        if not self.enabled:
            return
        try:
            with mlflow.start_run(
                experiment_id=self._experiment_id,
                run_name="model_load",
                tags={"event": "model_load"},
            ):
                mlflow.log_params(model_info)
                mlflow.log_metric("load_time_s", load_time_s)
        except Exception as e:
            logger.warning(f"MLflow model load logging failed: {e}")

    def get_recent_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch recent runs from MLflow."""
        if not self.enabled:
            return []
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self._experiment_id],
                order_by=["start_time DESC"],
                max_results=limit,
            )
            if runs.empty:
                return []
            return runs.to_dict(orient="records")
        except Exception as e:
            logger.warning(f"Failed to fetch MLflow runs: {e}")
            return []

    def get_experiment_stats(self) -> Dict[str, Any]:
        """Get aggregate stats for the experiment."""
        if not self.enabled:
            return {}
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self._experiment_id],
                max_results=1000,
            )
            if runs.empty:
                return {"total_runs": 0}

            stats = {
                "total_runs": len(runs),
                "avg_generation_time": runs.get("metrics.generation_time_s", runs.get(
                    "metrics.generation_time_s", None
                )),
            }
            if "metrics.generation_time_s" in runs.columns:
                stats["avg_generation_time_s"] = runs["metrics.generation_time_s"].mean()
                stats["min_generation_time_s"] = runs["metrics.generation_time_s"].min()
                stats["max_generation_time_s"] = runs["metrics.generation_time_s"].max()
            return stats
        except Exception as e:
            logger.warning(f"Failed to get experiment stats: {e}")
            return {}
