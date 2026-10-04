"""Package du projet.

Place les caches Hugging Face dans data/raw/hf_cache. Doit s'exécuter avant tout
import de datasets / huggingface_hub, qui lisent ces variables à l'import.
setdefault : une variable déjà définie (ex. sur la machine GPU) reste prioritaire.
"""
import os
from pathlib import Path

_CACHE = Path(__file__).resolve().parents[1] / "data" / "raw" / "hf_cache"
os.environ.setdefault("HF_HUB_CACHE", str(_CACHE / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(_CACHE / "datasets"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")