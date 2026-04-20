# app/utils/session.py
"""
Session State Management for Streamlit.
Centralized initialization and access for all session variables.
"""

from typing import Any, Optional, List, Dict
import streamlit as st
from PIL import Image


def init_session_state():
    """Initialize all session state variables with defaults."""

    defaults = {
        # Model state
        "model_loaded": False,
        "model_loading": False,
        "model_load_error": None,

        # Generation state
        "is_generating": False,
        "generation_progress": 0,
        "current_step": 0,
        "total_steps": 0,

        # Results
        "generated_images": [],          # List[Image.Image]
        "generation_history": [],        # List[dict] - last N results
        "selected_image_index": 0,
        "last_result": None,

        # Settings (persisted during session)
        "last_prompt": "",
        "last_negative_prompt": "",
        "last_seed": -1,

        # UI state
        "active_tab": "Generate",
        "gallery_page": 0,
        "show_advanced": False,

        # Tracking
        "session_generation_count": 0,
        "session_errors": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)


def set(key: str, value: Any):
    st.session_state[key] = value


def add_to_history(result_dict: Dict[str, Any], max_history: int = 50):
    """Add generation result to history."""
    history = st.session_state.get("generation_history", [])
    history.insert(0, result_dict)
    st.session_state["generation_history"] = history[:max_history]


def clear_history():
    st.session_state["generation_history"] = []
    st.session_state["generated_images"] = []
