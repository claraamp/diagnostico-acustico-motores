"""Coloca scripts/ no caminho de importação, como os próprios scripts fazem."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
