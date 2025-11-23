"""Site-wide customizations for 3DCAVLA utilities.

This module is automatically imported by Python (if present on sys.path)
before user code runs. We exploit this hook to provide a stub
`robosuite.macros_private` module, which robosuite tries to import for
user-specific overrides. We also redirect robosuite's default logging
location to a user-writable directory inside this repository so that we
don't need special permissions on /tmp.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path


def _ensure_robosuite_macros_private() -> None:
    """Register a lightweight `robosuite.macros_private` stub if missing."""
    module_name = "robosuite.macros_private"
    if module_name in sys.modules:
        return

    # Choose a log directory that is always writable by the current user.
    default_log_dir = Path.cwd() / "logs" / "robosuite"
    log_dir = Path(os.environ.get("ROBOSUITE_LOG_DIR", default_log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / os.environ.get("ROBOSUITE_LOG_FILE_NAME", "robosuite.log")

    module = types.ModuleType(module_name)
    module.__file__ = str(log_path)
    module.ROBOSUITE_LOG_DIR = str(log_dir)
    module.ROBOSUITE_LOG_PATH = str(log_path)
    module.ROBOSUITE_LOGGER_FILE = str(log_path)

    # Expose common path hints for robosuite callers that consult env vars.
    os.environ.setdefault("ROBOSUITE_LOG_DIR", str(log_dir))
    os.environ.setdefault("ROBOSUITE_LOG_PATH", str(log_path))

    sys.modules[module_name] = module


_ensure_robosuite_macros_private()
