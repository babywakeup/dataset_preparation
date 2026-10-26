import sys
from pathlib import Path


EXAMPLE_ROOT = str(Path(__file__).resolve().parents[1])
if EXAMPLE_ROOT not in sys.path:
    sys.path.insert(0, EXAMPLE_ROOT)
