"""Prepare the mixed system-Python/native-env runtime for CPU PyTorch."""

from __future__ import annotations

import sys
from pathlib import Path


def prepare_runtime() -> None:
    root = Path(__file__).resolve().parents[1]
    native_site = (root / ".conda-aligator" / "lib" / "python3.12" / "site-packages").resolve()
    src = str((root / "src").resolve())
    native_text = str(native_site)

    # The bundled NumPy and system PyTorch load different OpenMP runtimes on
    # macOS. Keep system NumPy/PyTorch, then add the bundled MuJoCo and
    # Gymnasium wheels after NumPy is cached.
    sys.path[:] = [entry for entry in sys.path if str(Path(entry or ".").resolve()) != native_text]
    if src not in sys.path:
        sys.path.insert(0, src)
    import numpy  # noqa: F401

    if native_text not in sys.path:
        sys.path.append(native_text)
