"""Make `fipd` importable whether or not the package has been pip-installed.

`pip install -e .` is the intended setup and makes this a no-op. This exists so
a freshly cloned repo runs immediately, which matters more for a student
project than import purity.
"""

import sys
from pathlib import Path

try:
    import fipd  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
