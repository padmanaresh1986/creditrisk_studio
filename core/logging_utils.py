from __future__ import annotations

import logging
from datetime import datetime
import traceback
from typing import Any

import streamlit as st

_LOGGER = logging.getLogger("creditrisk_studio")
if not _LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.DEBUG)
    _LOGGER.propagate = False


def _key_for_channel(channel: str) -> str:
    normalized = str(channel or "app").strip().lower()
    return {
        "prediction": "prediction_logs",
        "bulk": "bulk_logs",
        "training": "training_logs",
        "app": "app_logs",
    }.get(normalized, "app_logs")


def emit_log(message: str, level: str = "INFO", channel: str = "app") -> str:
    """Write a terminal log and retain a bounded UI-console copy in session state."""
    normalized_level = str(level or "INFO").upper()
    if normalized_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        normalized_level = "INFO"
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] [{normalized_level}] {message}"

    logger_method = getattr(_LOGGER, normalized_level.lower(), _LOGGER.info)
    logger_method(str(message))
    print(line, flush=True)

    key = _key_for_channel(channel)
    existing = list(st.session_state.get(key, []))
    st.session_state[key] = (existing + [line])[-500:]
    return line


def emit_exception(exc: BaseException, channel: str = "app", context: str = "Operation failed") -> str:
    """Log an exception summary to UI and full traceback to the terminal logger."""
    emit_log(f"{context}: {type(exc).__name__}: {exc}", level="ERROR", channel=channel)
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    _LOGGER.debug("%s\n%s", context, tb)
    print(tb, flush=True)
    return tb


def render_console(channel: str, title: str | None = None, expanded: bool = False) -> None:
    """Render a collapsed-by-default terminal-style console for a workflow channel."""
    key = _key_for_channel(channel)
    label = title or {
        "prediction": "Prediction console",
        "bulk": "Bulk prediction console",
        "training": "Processing console",
        "app": "Application console",
    }.get(str(channel).lower(), "Application console")
    logs = st.session_state.get(key, [])
    with st.expander(label, expanded=expanded):
        if not logs:
            st.caption("No logs recorded in this session.")
            return
        import html
        safe = html.escape("\n".join(logs))
        st.markdown(
            f"<div class='cr-console'><pre>{safe}</pre></div>",
            unsafe_allow_html=True,
        )
