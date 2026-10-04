"""Accès au registre des sources (configs/sources.yaml) et chargement épinglé.

Tous les scripts de données passent par load_source() : une source ne peut être
chargée qu'à la révision déclarée dans le registre (reproductibilité).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from datasets import DatasetDict, load_dataset

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "configs" / "sources.yaml"


def load_registry() -> dict[str, dict[str, Any]]:
    """Retourne le registre indexé par nom de source."""
    sources = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))["sources"]
    return {s["name"]: s for s in sources}


def load_source(name: str, config: str | None = None) -> DatasetDict:
    """Charge une source (et éventuellement une config) à sa révision épinglée."""
    source = load_registry()[name]
    revision = source.get("revision")
    if not revision:
        raise ValueError(f"Source '{name}' sans révision épinglée dans {REGISTRY_PATH.name}")
    return load_dataset(source["hf_id"], config, revision=revision)
