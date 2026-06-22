"""Demo 01 — EDA: panorama del dataset de SciPy.

Imprime el resumen ya calculado (data/processed/eda_scipy__scipy.json): tamaño, split
temporal, distribución de módulos y características del texto (código, tracebacks).

    python notebooks/01_eda.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.eda import render  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    report = json.loads((ROOT / "data" / "processed" / "eda_scipy__scipy.json").read_text(encoding="utf-8"))
    print(render(report))
    print("\n(recalcular desde cero: python -m src.data.eda scipy/scipy — requiere data/raw/)")


if __name__ == "__main__":
    main()
