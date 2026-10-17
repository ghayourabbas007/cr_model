"""ASGI entry point: `uvicorn forecaster.api.main:app` (or `forecaster serve`)."""

from forecaster.api.app import create_app
from forecaster.config import get_settings
from forecaster.logging_setup import configure_logging

_s = get_settings()
configure_logging(_s.log_level, _s.log_format)
app = create_app(_s)
