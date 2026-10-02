"""Prepare the mixed system-Python/native-env runtime for CPU PyTorch."""

from __future__ import annotations

import sys
from pathlib import Path


def prepare_runtime() -> None:
    root = Path(__file__).resolve().parents[1]
    native_site = root / ".conda-aligator" / "lib" / "python3.12" / "site-packages"
    if not native_site.is_dir():
        # The handoff packet keeps the native MuJoCo environment one directory
        # below the repository.  Prefer the repository-local environment when
        # it exists, but retain compatibility with the original layout.
        native_site = root / "gradient_cartpole_handoff" / ".conda-aligator" / "lib" / "python3.12" / "site-packages"
    native_site = native_site.resolve()
    src = str((root / "src").resolve())
    native_text = str(native_site)

    # Native Python 3.12 already resolves NumPy and MuJoCo from the bundled
    # environment.  Removing that site-packages directory before importing
    # NumPy leaves this interpreter with no numerical runtime at all.  The
    # path surgery below is only needed for the system-Python + Torch setup.
    if sys.version_info >= (3, 12):
        if src not in sys.path:
            sys.path.insert(0, src)
        # Keep system NumPy/PyTorch first, then make the matching Python 3.12
        # MuJoCo wheel importable. Prepending the native site-packages loads a
        # second NumPy/OpenMP runtime and can abort before training starts.
        import numpy  # noqa: F401

        if native_text not in sys.path:
            sys.path.append(native_text)
        return

    # The bundled NumPy and system PyTorch load different OpenMP runtimes on
    # macOS. Keep system NumPy/PyTorch, then add the bundled MuJoCo and
    # Gymnasium wheels after NumPy is cached.
    sys.path[:] = [entry for entry in sys.path if str(Path(entry or ".").resolve()) != native_text]
    if src not in sys.path:
        sys.path.insert(0, src)
    import numpy  # noqa: F401

    if native_text not in sys.path:
        sys.path.append(native_text)
