# app/main.py
"""
DreamForge — End-to-End Text-to-Image Studio
Stable Diffusion v1.5 with MLOps Pipeline

Run: streamlit run app/main.py
"""

import copy
import io
import sys
import time
import yaml
import zipfile
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from PIL import Image

from pipeline.model_manager import ModelManager, MODEL_ID
from pipeline.inference_engine import InferenceEngine, GenerationConfig
from pipeline.image_processor import ImageProcessor
from pipeline.prompt_processor import PromptProcessor
from monitoring.metrics_collector import MetricsCollector, GenerationMetric
from monitoring.mlflow_tracker import MLflowTracker
from app.utils.session import init_session_state, add_to_history
from app.components.sidebar import render_sidebar
from app.components.metrics_dashboard import render_mlops_dashboard


# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DreamForge — Stable Diffusion Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Marcellus&family=Montserrat:wght@300;400;600&display=swap');

  /* Global Theme */
  html, body, [class*="css"] {
    font-family: 'Montserrat', sans-serif;
    background-color: #0A0908;
    color: #F8F1E9;
  }

  h1, h2, h3 { 
    font-family: 'Marcellus', serif !important; 
    color: #D4AF37 !important;
    letter-spacing: 0.05em;
  }

  /* Background Glow */
  .stApp {
    background: radial-gradient(circle at 10% 20%, rgba(93, 16, 29, 0.15) 0%, #0A0908 40%),
                radial-gradient(circle at 90% 80%, rgba(255, 153, 51, 0.08) 0%, #0A0908 40%);
  }

  .stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background-image: url('https://www.transparenttextures.com/patterns/pinstriped-suit.png');
    opacity: 0.03;
    pointer-events: none;
  }

  .block-container { padding-top: 2rem !important; max-width: 1300px; }

  /* Premium Cards (Glassmorphism) */
  .gen-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(212, 175, 55, 0.15);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  }

  /* Image Frame with Golden Glow */
  .img-frame {
    border: 1px solid rgba(212, 175, 55, 0.4);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 0 25px rgba(212, 175, 55, 0.1);
    transition: all 0.3s ease;
  }
  .img-frame:hover {
    border-color: #D4AF37;
    transform: scale(1.01);
    box-shadow: 0 0 40px rgba(212, 175, 55, 0.2);
  }

  /* Royal Buttons */
  div.stButton > button {
    border-radius: 8px !important;
    font-family: 'Marcellus', serif !important;
    letter-spacing: 0.05em;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
  }

  div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #5D101D, #8B1A2C) !important;
    color: #F8F1E9 !important;
    border: 1px solid #D4AF37 !important;
    padding: 0.75rem 2.5rem !important;
    font-size: 1.1rem !important;
  }

  div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #8B1A2C, #B5263B) !important;
    box-shadow: 0 0 20px rgba(212, 175, 55, 0.3) !important;
    transform: translateY(-2px);
  }

  /* Styled Tabs */
  .stTabs [data-baseweb="tab-list"] {
    background: rgba(93, 16, 29, 0.1) !important;
    padding: 5px;
    border-radius: 12px;
    border: 1px solid rgba(212, 175, 55, 0.1);
  }
  .stTabs [data-baseweb="tab"] {
    color: #b0a898 !important;
    font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
    background: rgba(212, 175, 55, 0.1) !important;
    color: #D4AF37 !important;
    border-radius: 8px;
  }

  /* Metrics Custom Styling */
  [data-testid="metric-container"] {
    background: rgba(93, 16, 29, 0.1) !important;
    border: 1px solid rgba(212, 175, 55, 0.2) !important;
    border-radius: 12px;
    padding: 15px;
  }

  /* Inputs */
  .stTextArea textarea {
    background: rgba(0, 0, 0, 0.3) !important;
    border: 1px solid rgba(212, 175, 55, 0.2) !important;
    color: #F8F1E9 !important;
    border-radius: 12px !important;
  }
  .stTextArea textarea:focus {
    border-color: #D4AF37 !important;
    box-shadow: 0 0 10px rgba(212, 175, 55, 0.2) !important;
  }

  /* Sidebar Enhancements */
  section[data-testid="stSidebar"] {
    background: #0D0C0B !important;
    border-right: 1px solid rgba(212, 175, 55, 0.15) !important;
  }

  /* Progress Bar */
  .stProgress > div > div > div > div {
    background: linear-gradient(90deg, #5D101D, #FF9933) !important;
  }

  /* Status Pills */
  .status-ready { background: rgba(0, 91, 150, 0.2); color: #74b9ff; border: 1px solid rgba(0, 91, 150, 0.4); }

  /* Status Bar */
  .status-bar {
    display: flex;
    gap: 12px;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
  }
  .status-item {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(212, 175, 55, 0.2);
    border-radius: 10px;
    padding: 6px 14px;
    font-size: 0.82rem;
    display: flex;
    align-items: center;
    gap: 8px;
    backdrop-filter: blur(4px);
  }
  .status-label {
    color: rgba(212, 175, 55, 0.7);
    text-transform: uppercase;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.05em;
  }
  .status-value {
    color: #F8F1E9;
    font-weight: 600;
  }
  .pulse-dot {
    width: 8px;
    height: 8px;
    background: #4caf50;
    border-radius: 50%;
    box-shadow: 0 0 10px #4caf50;
    animation: pulse 2s infinite;
  }
  .pulse-dot-white {
    width: 8px;
    height: 8px;
    background: #ffffff;
    border-radius: 50%;
    box-shadow: 0 0 10px #ffffff;
    animation: pulse-white 2s infinite;
  }
  @keyframes pulse {
    0% { transform: scale(0.95); opacity: 0.7; }
    50% { transform: scale(1.05); opacity: 1; box-shadow: 0 0 15px #4caf50; }
    100% { transform: scale(0.95); opacity: 0.7; }
  }
  @keyframes pulse-white {
    0% { transform: scale(0.95); opacity: 0.7; }
    50% { transform: scale(1.05); opacity: 1; box-shadow: 0 0 15px #ffffff; }
    100% { transform: scale(0.95); opacity: 0.7; }
  }
  .status-loading { background: rgba(255,152,0,0.2); color: #ffb74d; border: 1px solid rgba(255,152,0,0.3); }
  .status-error { background: rgba(244,67,54,0.2); color: #ef9a9a; border: 1px solid rgba(244,67,54,0.3); }
  .status-ready { background: rgba(0, 91, 150, 0.2); color: #74b9ff; border: 1px solid rgba(0, 91, 150, 0.4); }

  /* Hide Streamlit Decorations (Keeping sidebar toggle) */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  [data-testid="stHeader"] {background: rgba(0,0,0,0) !important;}
  .stDeployButton {display:none !important;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_config():
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


# ─────────────────────────────────────────────────────────────────────
# SINGLETONS (cached across reruns)
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_inference_engine() -> InferenceEngine:
    return InferenceEngine()


@st.cache_resource
def get_image_processor(output_dir: str) -> ImageProcessor:
    return ImageProcessor(output_dir=output_dir)


@st.cache_resource
def get_metrics_collector(port: int = 8012, enable: bool = True, window_size: int = 100) -> MetricsCollector:
    return MetricsCollector(
        prometheus_port=port,
        enable_prometheus=enable,
        window_size=window_size
    )


@st.cache_resource
def get_mlflow_tracker(tracking_uri: str, experiment_name: str, artifact_location: str = None) -> MLflowTracker:
    return MLflowTracker(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        artifact_location=artifact_location
    )


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
def model_status_html(loaded: bool, loading: bool, error: str | None) -> str:
    if loading:
        return "<span class='status-pill status-loading'>⏳ Loading Engine...</span>"
    if error:
        return f"<span class='status-pill status-error'>⚠️ Engine Error</span>"
    if loaded:
        return "<span class='status-pill status-ready'><span class='pulse-dot' style='display:inline-block; margin-right:5px;'></span> Engine Ready</span>"
    return "<span class='status-pill status-loading'><span class='pulse-dot-white' style='display:inline-block; margin-right:5px;'></span> Idle</span>"


def image_to_download_bytes(images: list[Image.Image]) -> bytes:
    """Create a zip of all generated images for download."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, img in enumerate(images):
            img_buf = io.BytesIO()
            img.save(img_buf, format="PNG")
            zf.writestr(f"image_{i + 1}.png", img_buf.getvalue())
    return buf.getvalue()


def record_generation_metrics(result, gen_config, metrics, mlflow_tracker, settings):
    """Helper to record metrics to Prometheus and MLflow."""
    # 1. Prometheus / In-memory
    metric = GenerationMetric(
        request_id=result.request_id,
        timestamp=result.timestamp,
        generation_time_s=result.generation_time_s,
        steps=gen_config.num_inference_steps,
        width=gen_config.width,
        height=gen_config.height,
        num_images=gen_config.num_images,
        scheduler=gen_config.scheduler,
        success=result.success,
        error=result.error if not result.success else None,
    )
    metrics.record_generation(metric)

    # 2. MLflow
    if result.success and settings.get("log_to_mlflow", True):
        mlflow_tracker.log_generation(
            prompt=result.prompt_used,
            negative_prompt=result.negative_prompt_used,
            config={
                "width": gen_config.width,
                "height": gen_config.height,
                "num_inference_steps": gen_config.num_inference_steps,
                "guidance_scale": gen_config.guidance_scale,
                "scheduler": gen_config.scheduler,
                "style_preset": gen_config.style_preset,
                "num_images": gen_config.num_images,
                "quality_boost": gen_config.quality_boost,
            },
            metrics={
                "generation_time_s": result.generation_time_s,
                "steps_per_second": gen_config.num_inference_steps / result.generation_time_s if result.generation_time_s > 0 else 0
            },
            images=result.images,
            seed=result.seed_used,
            request_id=result.request_id,
        )


# ─────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────
def main():
    config = load_config()
    init_session_state()

    mlops_cfg = config.get("mlops", {})
    model_cfg = config.get("model", {})
    storage_cfg = config.get("storage", {})

    engine = get_inference_engine()
    image_processor = get_image_processor(storage_cfg.get("output_dir", "./artifacts/generated"))
    
    # MLops Tools configuration
    monitor_cfg = mlops_cfg.get("monitoring", {})
    metrics = get_metrics_collector(
        port=monitor_cfg.get("prometheus_port", 8012),
        enable=monitor_cfg.get("enable_prometheus", True),
        window_size=monitor_cfg.get("metrics_window", 100)
    )
    
    mf_cfg = mlops_cfg.get("mlflow", {})
    mlflow_tracker = get_mlflow_tracker(
        tracking_uri=mf_cfg.get("tracking_uri", "http://localhost:5012"),
        experiment_name=mf_cfg.get("experiment_name", "stable-diffusion-v1-5"),
        artifact_location=mf_cfg.get("artifact_location", None)
    )

    # ── SIDEBAR ────────────────────────────────────────────────────────
    settings = render_sidebar(config)

    # Sync scheduler with model manager in real-time
    if engine.model_manager.is_loaded():
        current_scheduler = engine.model_manager.model_info.scheduler
        if current_scheduler != settings["scheduler"]:
            engine.model_manager.set_scheduler(settings["scheduler"])

    # ── HEADER ─────────────────────────────────────────────────────────
    header_col, status_col = st.columns([4, 1])
    with header_col:
        st.markdown("""
        <h1 style='margin:0; font-size:2.2rem; background: linear-gradient(135deg, #e8a87c, #c9883c);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            🎨 DreamForge Studio
        </h1>
        """, unsafe_allow_html=True)

    with status_col:
        loaded = engine.model_manager.is_loaded()
        loading = st.session_state.get("model_loading", False)
        error = st.session_state.get("model_load_error", None)
        # Status indication (top right)
        st.markdown(f"""
            <div style='text-align:right; padding-top:1.5rem;'>
                <span style='color:rgba(212, 175, 55, 0.6); font-size:0.65rem; font-weight:700; display:block; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.05em;'>System Health</span>
                {model_status_html(loaded, loading, error)}
            </div>
            """, unsafe_allow_html=True)

    # ── TABS ───────────────────────────────────────────────────────────
    tab_gen, tab_gallery, tab_mlops, tab_about = st.tabs([
        "🖼 Generate",
        "🗂 Gallery",
        "📊 MLOps",
        "ℹ️ About",
    ])

    # ══════════════════════════════════════════════════════════════════
    # TAB 1 — GENERATE
    # ══════════════════════════════════════════════════════════════════
    with tab_gen:
        # Model loading panel (shown when not loaded)
        if not engine.model_manager.is_loaded():
            with st.container():
                st.markdown("""
                <div class='gen-card'>
                    <h3 style='margin-top:0;'>🚀 Load Model</h3>
                    <p style='color:#888;'>Load Stable Diffusion v1.5 to start generating images.</p>
                </div>
                """, unsafe_allow_html=True)

                info_col, btn_col = st.columns([3, 1])
                with info_col:
                    st.markdown(f"""
                    - **Model**: `{MODEL_ID}`
                    - **Device**: Auto-detected ({engine.model_manager.get_device('auto')})
                    """)

                with btn_col:
                    if st.button(
                        "⚡ Load Model",
                        type="primary",
                        disabled=st.session_state.get("model_loading", False),
                        use_container_width=True,
                    ):
                        st.session_state["model_loading"] = True
                        st.session_state["model_load_error"] = None

                if st.session_state.get("model_loading", False):
                    load_placeholder = st.empty()
                    load_placeholder.info("⏳ Loading model... This may take several minutes on first run.")
                    progress_bar = st.progress(0, text="Initializing...")

                    try:
                        start = time.time()
                        progress_bar.progress(10, "Downloading / loading weights...")

                        load_cfg = config.get("model", {})
                        engine.model_manager.load_model(
                            model_id=MODEL_ID,
                            device=load_cfg.get("device", "auto"),
                            dtype=load_cfg.get("dtype", "float16"),
                            cache_dir=load_cfg.get("cache_dir", "./model_cache"),
                            enable_xformers=load_cfg.get("enable_xformers", True),
                            enable_attention_slicing=load_cfg.get("enable_attention_slicing", True),
                            enable_vae_slicing=load_cfg.get("enable_vae_slicing", True),
                            safety_checker=load_cfg.get("safety_checker", True),
                            scheduler=settings["scheduler"],
                        )

                        progress_bar.progress(100, "Model ready!")
                        load_time = time.time() - start
                        st.session_state["model_loading"] = False
                        st.session_state["model_loaded"] = True
                        load_placeholder.success(f"✅ Model loaded in {load_time:.1f}s")

                        mlflow_tracker.log_model_load(
                            {"model_id": MODEL_ID, "device": engine.model_manager.model_info.device},
                            load_time_s=load_time,
                        )
                        st.rerun()

                    except Exception as e:
                        st.session_state["model_loading"] = False
                        st.session_state["model_load_error"] = str(e)
                        load_placeholder.error(f"❌ Failed to load model: {e}")
                        progress_bar.empty()

        # ── GENERATION FORM (shown when model loaded) ──────────────────
        else:
            model_info = engine.model_manager.get_model_info()

            # Show modern status bar
            if model_info:
                st.markdown(f"""
                <div class="status-bar">
                    <div class="status-item">
                        <div>
                            <span class="status-label">Model Engine</span><br>
                            <span class="status-value">{model_info.model_id.split('/')[-1]}</span>
                        </div>
                    </div>
                    <div class="status-item">
                        <span class="status-label">Hardware</span><br>
                        <span class="status-value">{model_info.device.upper()}</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">Algorithm</span><br>
                        <span class="status-value">{model_info.scheduler}</span>
                    </div>
                """, unsafe_allow_html=True)

            # Prompt inputs
            prompt_col, neg_col = st.columns([3, 2])
            with prompt_col:
                st.markdown("#### Prompt")
                prompt = st.text_area(
                    "Describe your image",
                    value=st.session_state.get("last_prompt", ""),
                    height=110,
                    placeholder="A majestic dragon soaring over misty mountains at golden hour, epic fantasy art...",
                    label_visibility="collapsed",
                )
                # Example prompts
                with st.expander("💡 Example Prompts"):
                    examples = PromptProcessor.get_example_prompts()
                    cols = st.columns(2)
                    for i, ex in enumerate(examples):
                        with cols[i % 2]:
                            if st.button(
                                f"↗ {ex[:45]}...",
                                key=f"ex_{i}",
                                use_container_width=True,
                            ):
                                st.session_state["last_prompt"] = ex
                                st.rerun()

            with neg_col:
                st.markdown("#### Negative Prompt")
                negative_prompt = st.text_area(
                    "What to exclude",
                    value="",
                    height=110,
                    placeholder="blurry, bad anatomy, extra limbs...",
                    label_visibility="collapsed",
                )

            # Enhanced prompt preview
            if settings.get("show_enhanced_prompt") and prompt:
                pp = PromptProcessor()
                processed = pp.process(
                    prompt=prompt,
                    style_preset=settings["style_preset"],
                    quality_boost=settings["quality_boost"],
                )
                if processed.enhanced != prompt:
                    st.caption(f"✨ **Enhanced:** {processed.enhanced[:200]}...")

                for warn in processed.warnings:
                    st.warning(warn)

            # Generate button
            st.markdown("")
            btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 1])
            with btn_col1:
                generate_clicked = st.button(
                    "🎨 Generate Image",
                    type="primary",
                    disabled=not prompt.strip() or st.session_state.get("is_generating", False),
                    use_container_width=True,
                )

            with btn_col2:
                if st.button("🗑 Clear Results", use_container_width=True):
                    st.session_state["generated_images"] = []
                    st.rerun()

            with btn_col3:
                if st.button("🔄 Unload Model", use_container_width=True):
                    engine.model_manager.unload_model()
                    st.session_state["model_loaded"] = False
                    st.rerun()

            # ── GENERATION ─────────────────────────────────────────────
            if generate_clicked and prompt.strip():
                st.session_state["is_generating"] = True
                st.session_state["last_prompt"] = prompt

                gen_config = GenerationConfig(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=settings["width"],
                    height=settings["height"],
                    num_inference_steps=settings["num_inference_steps"],
                    guidance_scale=settings["guidance_scale"],
                    seed=settings["seed"] if settings["seed"] != -1 else None,
                    num_images=settings["num_images"],
                    scheduler=settings["scheduler"],
                    style_preset=settings["style_preset"],
                    negative_preset=settings["negative_preset"],
                    quality_boost=settings["quality_boost"],
                )

                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                prog_bar = st.progress(0)

                def progress_cb(step: int, total: int, _latents):
                    pct = int((step / total) * 100)
                    prog_bar.progress(pct, f"Step {step}/{total}")
                    status_placeholder.caption(
                        f"⚙️ Generating... step {step}/{total} ({pct}%)"
                    )

                metrics.set_active_generations(1)

                # Check if we need to do comparison
                lora_mode = settings.get("lora_mode", "Base Model (Default)")

                if lora_mode == "Comparison (Side-by-Side)":
                    status_placeholder.info("🧪 Comparison mode: Generating original vs. fine-tuned...")

                    # 1. Base Model (ensure LoRA is unloaded)
                    engine.model_manager.unload_lora()
                    result_base = engine.generate(gen_config, progress_callback=progress_cb)
                    record_generation_metrics(result_base, gen_config, metrics, mlflow_tracker, settings)

                    if result_base.success:
                        # 2. Fine-tuned Model (load LoRA)
                        status_placeholder.info("🧬 Applying LoRA fine-tuning...")
                        engine.model_manager.load_lora("lora_weights/best")

                        # Use same seed for exact comparison — copy to avoid mutating gen_config
                        gen_config_lora = copy.copy(gen_config)
                        gen_config_lora.seed = result_base.seed_used

                        result_lora = engine.generate(gen_config_lora, progress_callback=progress_cb)
                        record_generation_metrics(result_lora, gen_config_lora, metrics, mlflow_tracker, settings)

                        # Combine results
                        if result_lora.success:
                            result = result_lora
                            result.images = [result_base.images[0], result_lora.images[0]]
                        else:
                            result = result_lora
                    else:
                        result = result_base

                    # 3. Cleanup: Unload LoRA to keep base model as default (optional, but safer)
                    engine.model_manager.unload_lora()

                elif lora_mode == "Fine-tuned (LoRA)":
                    # Load LoRA if not already loaded
                    if engine.model_manager.current_lora_path != "lora_weights/best":
                        status_placeholder.info("🧬 Loading LoRA fine-tuning...")
                        engine.model_manager.load_lora("lora_weights/best")
                    
                    result = engine.generate(gen_config, progress_callback=progress_cb)
                else:
                    # Base Model: Unload LoRA if currently loaded
                    if engine.model_manager.current_lora_path is not None:
                        status_placeholder.info("🧹 Reverting to base model...")
                        engine.model_manager.unload_lora()
                    
                    result = engine.generate(gen_config, progress_callback=progress_cb)

                metrics.set_active_generations(0)

                prog_bar.empty()
                status_placeholder.empty()
                progress_placeholder.empty()
                st.session_state["is_generating"] = False

                if result.success:
                    st.session_state["generated_images"] = result.images
                    st.session_state["last_result"] = result
                    st.session_state["session_generation_count"] = (
                        st.session_state.get("session_generation_count", 0) + 1
                    )

                    # For non-comparison modes, we record metrics here.
                    # Comparison mode already recorded them twice inside its branch.
                    if lora_mode != "Comparison (Side-by-Side)":
                        record_generation_metrics(result, gen_config, metrics, mlflow_tracker, settings)

                    # Add to history (common for all modes)
                    add_to_history({
                        "request_id": result.request_id,
                        "prompt": prompt[:100],
                        "generation_time_s": result.generation_time_s,
                        "steps_per_second": gen_config.num_inference_steps / result.generation_time_s if result.generation_time_s > 0 else 0,
                        "steps": gen_config.num_inference_steps,
                        "width": gen_config.width,
                        "height": gen_config.height,
                        "num_images": len(result.images),
                        "scheduler": gen_config.scheduler,
                        "seed": result.seed_used,
                        "success": True,
                    })

                    # Save to disk
                    if settings.get("save_to_disk", True):
                        for i, img in enumerate(result.images):
                            image_processor.save_image(
                                image=img,
                                request_id=result.request_id,
                                index=i,
                                metadata={
                                    "prompt": result.prompt_used,
                                    "negative_prompt": result.negative_prompt_used,
                                    "seed": result.seed_used,
                                    "steps": gen_config.num_inference_steps,
                                    "guidance_scale": gen_config.guidance_scale,
                                    "width": gen_config.width,
                                    "height": gen_config.height,
                                    "scheduler": gen_config.scheduler,
                                    "generation_time_s": result.generation_time_s,
                                    "request_id": result.request_id,
                                },
                            )

                    success_msg = (
                        f"✅ Generated in **{result.generation_time_s:.1f}s** "
                        f"(seed: `{result.seed_used}`)"
                    )
                    if result.nsfw_detected:
                        success_msg += " ⚠️ NSFW content detected and filtered."
                    st.success(success_msg)

                else:
                    st.session_state["session_errors"] = (
                        st.session_state.get("session_errors", 0) + 1
                    )
                    if lora_mode != "Comparison (Side-by-Side)":
                        metrics.record_generation(GenerationMetric(
                            request_id=result.request_id,
                            timestamp=result.timestamp,
                            generation_time_s=result.generation_time_s,
                            steps=gen_config.num_inference_steps,
                            width=gen_config.width,
                            height=gen_config.height,
                            num_images=gen_config.num_images,
                            scheduler=gen_config.scheduler,
                            success=False,
                            error=result.error,
                        ))
                    st.error(f"❌ Generation failed: {result.error}")

            # ── DISPLAY RESULTS ─────────────────────────────────────────
            generated_images = st.session_state.get("generated_images", [])
            if generated_images:
                st.markdown("---")
                st.markdown("### Generated Images")

                last_result = st.session_state.get("last_result")

                lora_mode = settings.get("lora_mode")
                if len(generated_images) == 2 and lora_mode == "Comparison (Side-by-Side)":
                    col_base, col_lora = st.columns(2)
                    with col_base:
                        st.markdown("<h4 style='text-align:center;'>Original Model (SD v1.5)</h4>", unsafe_allow_html=True)
                        st.image(generated_images[0], use_container_width=True, caption="Base Weights")
                    with col_lora:
                        st.markdown("<h4 style='text-align:center; color:#e8a87c;'>Fine-tuned Model (Indian Festivals)</h4>", unsafe_allow_html=True)
                        st.image(generated_images[1], use_container_width=True, caption="LoRA Weights Applied")

                    st.divider()
                    st.markdown("#### 📋 Comparison Details")
                    st.markdown(f"**Prompt**: `{last_result.prompt_used}`")
                    st.markdown(f"**Seed**: `{last_result.seed_used}` | **Steps**: {last_result.config.num_inference_steps} | **CFG**: {last_result.config.guidance_scale}")

                elif len(generated_images) == 1:
                    img_col, meta_col = st.columns([2, 1])
                    with img_col:
                        st.image(
                            generated_images[0],
                            use_container_width=True,
                            caption=f"Seed: {last_result.seed_used if last_result else 'unknown'}",
                        )
                    with meta_col:
                        if last_result:
                            st.markdown("#### 📋 Generation Details")
                            st.markdown(f"""
                            | Parameter | Value |
                            |-----------|-------|
                            | Seed | `{last_result.seed_used}` |
                            | Steps | {last_result.config.num_inference_steps} |
                            | CFG Scale | {last_result.config.guidance_scale} |
                            | Size | {last_result.config.width}×{last_result.config.height} |
                            | Scheduler | {last_result.config.scheduler} |
                            | Time | {last_result.generation_time_s:.2f}s |
                            | Speed | {last_result.config.num_inference_steps / last_result.generation_time_s:.1f} steps/s |
                            """)

                            st.markdown("**Prompt used:**")
                            st.code(last_result.prompt_used[:300], language=None)

                            # Download button
                            img_bytes = image_processor.image_to_bytes(generated_images[0])
                            st.download_button(
                                "⬇️ Download PNG",
                                data=img_bytes,
                                file_name=f"dreamforge_{last_result.seed_used}.png",
                                mime="image/png",
                                use_container_width=True,
                            )
                else:
                    # Grid display for multiple images
                    n_cols = min(len(generated_images), 2)
                    cols = st.columns(n_cols)
                    for i, img in enumerate(generated_images):
                        with cols[i % n_cols]:
                            st.image(img, use_container_width=True, caption=f"Image {i + 1}")

                    # Download all as zip
                    zip_bytes = image_to_download_bytes(generated_images)
                    st.download_button(
                        f"⬇️ Download All {len(generated_images)} Images (.zip)",
                        data=zip_bytes,
                        file_name=f"dreamforge_batch.zip",
                        mime="application/zip",
                    )

    # ══════════════════════════════════════════════════════════════════
    # TAB 2 — GALLERY
    # ══════════════════════════════════════════════════════════════════
    with tab_gallery:
        st.markdown("## 🗂 Session Gallery")

        history = st.session_state.get("generation_history", [])
        if not history:
            st.info("🎨 Generate some images to see them here!")
        else:
            st.markdown(f"**{len(history)} generations this session**")

            for i, h in enumerate(history[:20]):
                with st.expander(
                    f"🖼 {h.get('prompt', 'No prompt')[:60]}... "
                    f"— {h.get('generation_time_s', 0):.1f}s | seed {h.get('seed', '?')}",
                    expanded=(i == 0),
                ):
                    detail_col1, detail_col2 = st.columns([2, 1])
                    with detail_col2:
                        st.markdown(f"""
                        - **Steps**: {h.get('steps', '?')}
                        - **Size**: {h.get('width', '?')}×{h.get('height', '?')}
                        - **Scheduler**: {h.get('scheduler', '?')}
                        - **Seed**: `{h.get('seed', '?')}`
                        - **Time**: {h.get('generation_time_s', 0):.2f}s
                        - **Speed**: {h.get('steps_per_second', 0):.1f} steps/s
                        """)

    # ══════════════════════════════════════════════════════════════════
    # TAB 3 — MLOPS
    # ══════════════════════════════════════════════════════════════════
    with tab_mlops:
        stats = metrics.get_rolling_stats()
        gpu_stats = metrics.get_gpu_stats()
        history_data = metrics.get_recent_history(50)

        render_mlops_dashboard(
            stats=stats,
            gpu_stats=gpu_stats,
            history=history_data,
            mlflow_enabled=mlflow_tracker.enabled,
            mlflow_tracking_uri=mf_cfg.get("tracking_uri", "http://localhost:5012"),
            prometheus_enabled=monitor_cfg.get("enable_prometheus", True),
            prometheus_ui_url=monitor_cfg.get("prometheus_ui_url", "http://localhost:9092"),
        )

        # Auto-refresh toggle
        st.divider()
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔄 Refresh Metrics", use_container_width=True):
                st.rerun()
        with col2:
            st.caption("Metrics update after each generation. Click refresh for latest view.")

    # ══════════════════════════════════════════════════════════════════
    # TAB 4 — ABOUT
    # ══════════════════════════════════════════════════════════════════
    with tab_about:
        st.markdown("""
        ## ℹ️ DreamForge — SD v1.5 MLOps Studio

        ### Architecture

        ```
        User Input (Streamlit)
              │
              ▼
        PromptProcessor ──── Style Presets, Quality Boost, Validation
              │
              ▼
        InferenceEngine ──── Seed Management, Progress Tracking, Error Handling
              │
              ▼
        SD v1.5 Pipeline ──── Scheduler, VAE, UNet, CLIP Text Encoder
              │
              ▼
        ImageProcessor ────── Metadata Embedding, Storage, Grid Layout
              │
              ▼
        MetricsCollector ──── Latency, Throughput, Error Rate, GPU Stats
              │
              ▼
        MLflowTracker ──────── Experiment Logging, Artifact Storage, Run History
        ```

        ### Model
        - **Model**: [stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
        - **Architecture**: Latent Diffusion Model (LDM)
        - **Text Encoder**: CLIP ViT-L/14
        - **Resolution**: Native 512×512 (up to 768×768)

        ### Schedulers
        | Scheduler | Best For |
        |-----------|----------|
        | DDIM | Fast, deterministic, good quality |
        | Euler A | Varied, creative outputs |
        | DPM++ 2M | High quality, fewer steps needed |
        | PNDM | Stable baseline |
        | LMS | Smooth, classical |

        ### MLOps Pipeline
        - **MLflow**: Experiment tracking, parameter logging, image artifacts
        - **Prometheus**: Real-time metrics export (port 8012)
        - **In-memory**: Rolling stats, latency percentiles, throughput
        - **Disk**: PNG + JSON sidecar files with embedded metadata

        ### Memory Tips
        - Use 512×512 for lowest VRAM (4GB min)
        - Enable attention slicing for <6GB VRAM
        - Float16 reduces memory by ~2× vs float32

        ---
        """)


if __name__ == "__main__":
    main()