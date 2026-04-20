# app/components/sidebar.py
"""
Sidebar Component: All generation settings and controls.
"""

import streamlit as st
from pipeline.prompt_processor import PromptProcessor


def render_sidebar(config: dict) -> dict:
    """
    Render the full settings sidebar.

    Returns:
        dict with all current generation settings
    """
    inference_cfg = config.get("inference", {})
    model_cfg = config.get("model", {})

    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 8px 0 16px;'>
            <h2 style='margin:0; font-family: "Marcellus", serif; color: #D4AF37; letter-spacing: 0.15em;'>DreamForge</h2>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Model Section ──────────────────────────────────────────────
        st.markdown("### ☸️ Model")

        scheduler = st.selectbox(
            "Scheduler",
            options=inference_cfg.get("schedulers", ["DDIM", "PNDM", "Euler", "Euler A", "DPM++ 2M"]),
            index=0,
            help="Noise scheduler — DDIM is stable, Euler A adds variety, DPM++ 2M is high quality",
        )

        lora_mode = st.selectbox(
            "Model Type",
            options=["Base Model (Default)", "Fine-tuned (LoRA)"],
            index=0,
            help="Select which model weights to use for generation.",
        )

        # ── Image Settings ──────────────────────────────────────────────
        st.markdown("### 🪔 Image Canvas")

        col1, col2 = st.columns(2)
        with col1:
            width = st.select_slider(
                "Width",
                options=[256, 384, 512, 640, 768],
                value=512,
            )
        with col2:
            height = st.select_slider(
                "Height",
                options=[256, 384, 512, 640, 768],
                value=512,
            )

        num_images = st.slider(
            "Number of Images",
            min_value=1,
            max_value=inference_cfg.get("max_batch_size", 4),
            value=1,
            help="Generate multiple variations at once",
        )

        # ── Generation Parameters ───────────────────────────────────────
        st.markdown("### 🏵️ Parameters")

        steps = st.slider(
            "Inference Steps",
            min_value=inference_cfg.get("min_steps", 1),
            max_value=inference_cfg.get("max_steps", 150),
            value=inference_cfg.get("default_steps", 20),
            help="More steps = higher quality but slower. 20-30 is usually good.",
        )

        guidance_scale = st.slider(
            "CFG Scale (Guidance)",
            min_value=1.0,
            max_value=20.0,
            value=inference_cfg.get("default_guidance_scale", 7.5),
            step=0.5,
            help="How closely to follow the prompt. 7-8 is typical.",
        )

        seed = st.number_input(
            "Seed (-1 = random)",
            min_value=-1,
            max_value=2**31 - 1,
            value=-1,
            help="Use a fixed seed to reproduce results",
        )

        # ── Prompt Enhancement ──────────────────────────────────────────
        st.markdown("### 🏮 Enhancement")

        style_preset = st.selectbox(
            "Style Preset",
            options=PromptProcessor.get_style_presets(),
            index=0,
            help="Add style modifiers to your prompt automatically",
        )

        negative_preset = st.selectbox(
            "Negative Preset",
            options=PromptProcessor.get_negative_presets(),
            index=0,
            help="Pre-built negative prompt templates",
        )

        quality_boost = st.toggle(
            "Quality Boost",
            value=False,
            help="Append quality tokens: masterpiece, best quality, etc.",
        )

        # ── Advanced ────────────────────────────────────────────────────
        with st.expander("🔧 Advanced / Debug"):
            show_enhanced_prompt = st.checkbox("Show enhanced prompt", value=True)
            save_to_disk = st.checkbox("Save images to disk", value=True)
            log_to_mlflow = st.checkbox("Log to MLflow", value=True)

        st.divider()

        # Quick stats
        if st.session_state.get("session_generation_count", 0) > 0:
            gen_count = st.session_state.get("session_generation_count", 0)
            err_count = st.session_state.get("session_errors", 0)
            st.markdown(f"""
            <div style='font-size:0.8rem; color:#888; text-align:center;'>
                Session: {gen_count} generations · {err_count} errors
            </div>
            """, unsafe_allow_html=True)

    return {
        "scheduler": scheduler,
        "width": width,
        "height": height,
        "num_images": num_images,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "style_preset": style_preset,
        "negative_preset": negative_preset,
        "quality_boost": quality_boost,
        "show_enhanced_prompt": show_enhanced_prompt,
        "save_to_disk": save_to_disk,
        "log_to_mlflow": log_to_mlflow,
        "lora_mode": lora_mode,
    }
