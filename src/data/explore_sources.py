"""Exploration des sources de données déclarées dans configs/sources.yaml.

Pour chaque source (et chaque config/subset), on relève : splits, volumétrie,
colonnes et types, longueurs de texte, un exemple tronqué, la licence déclarée
sur le Hub et la révision (commit sha) exacte téléchargée.

Sorties (traçabilité) :
    reports/data_inventory.json   -> machine-readable
    reports/data_inventory.md     -> lisible, réutilisable dans le rapport

Usage (depuis la racine du projet) :
    python -m src.data.explore_sources
    python -m src.data.explore_sources --only medquad
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "sources.yaml"
REPORT_DIR = ROOT / "reports"

MAX_CHARS = 300        # troncature des exemples affichés
STATS_SAMPLE = 1000    # nb de lignes utilisées pour les stats de longueur
TEXT_DTYPES = {"string", "large_string"}

SEED = 42  # pour reproductibilité des stats de longueur


def truncate(value: Any) -> Any:
    """Tronque récursivement un exemple pour qu'il reste lisible."""
    if isinstance(value, str):
        return value if len(value) <= MAX_CHARS else value[:MAX_CHARS] + "…"
    if isinstance(value, list):
        return [truncate(v) for v in value[:5]]
    if isinstance(value, dict):
        return {k: truncate(v) for k, v in value.items()}
    return value


def text_length_stats(split) -> dict[str, dict[str, int]]:
    """Médiane et max (en caractères) des colonnes texte, sur un échantillon."""
    sample = split.shuffle(seed=SEED).select(range(min(STATS_SAMPLE, len(split))))
    stats = {}
    for col, feature in split.features.items():
        if getattr(feature, "dtype", None) in TEXT_DTYPES:
            lengths = [len(x) for x in sample[col] if isinstance(x, str)]
            if lengths:
                stats[col] = {"median": int(statistics.median(lengths)), "max": max(lengths)}
    return stats


def hub_metadata(hf_id: str) -> dict[str, Any]:
    """Licence déclarée et révision exacte : base du versionnage des sources."""
    try:
        info = HfApi().dataset_info(hf_id)
        licence = getattr(info.card_data, "license", None) if info.card_data else None
        return {"revision": info.sha, "licence_card": licence, "gated": info.gated}
    except Exception as exc:  # réseau, dataset privé, etc.
        return {"revision": None, "licence_card": None, "error": repr(exc)}


def explore_source(source: dict[str, Any]) -> dict[str, Any]:
    hf_id = source["hf_id"]
    result: dict[str, Any] = {**source, **hub_metadata(hf_id), "configs": {}}

    try:
        configs = source.get("configs") or get_dataset_config_names(hf_id)
    except Exception as exc:
        result["error"] = f"configs: {exc!r}"
        return result

    for cfg in configs:
        print(f"  - config '{cfg}'…")
        try:
            # On épingle la révision : relancer le script redonne les mêmes données.
            ds = load_dataset(hf_id, cfg, revision=result.get("revision"))

        except Exception as exc:
            result["configs"][cfg] = {"error": repr(exc)}
            continue

        splits = {}
        for split_name, split in ds.items():
            splits[split_name] = {
                "num_rows": len(split),
                "features": {k: str(v) for k, v in split.features.items()},
                "text_lengths": text_length_stats(split),
                "example": truncate(split[0]) if len(split) else None,
            }
        result["configs"][cfg] = {"splits": splits}
    return result


def to_markdown(inventory: dict[str, Any]) -> str:
    lines = [f"# Inventaire des sources — {inventory['generated_at']}", ""]
    for src in inventory["sources"]:
        lines += [f"## {src['name']} (`{src['hf_id']}`)", "",
                  f"- Langue : {src.get('langue')} — rôle prévu : {src.get('role_prevu')}",
                  f"- Licence (card) : {src.get('licence_card')} — révision : `{src.get('revision')}`"]
        if src.get("error"):
            lines.append(f"- **Erreur** : {src['error']}")
        for cfg, content in src["configs"].items():
            lines += ["", f"### Config `{cfg}`"]
            if "error" in content:
                lines.append(f"**Erreur** : {content['error']}")
                continue
            for split_name, info in content["splits"].items():
                lines += ["", f"**{split_name}** — {info['num_rows']} lignes",
                          "", "Colonnes : " + ", ".join(f"`{c}`" for c in info["features"]),
                          "", "Longueurs (car., médiane / max) : " + ", ".join(
                              f"`{c}` {s['median']} / {s['max']}"
                              for c, s in info["text_lengths"].items()),
                          "", "```json",
                          json.dumps(info["example"], ensure_ascii=False, indent=2),
                          "```"]
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", help="nom d'une source à explorer seule")
    args = parser.parse_args()

    sources = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["sources"]
    if args.only:
        sources = [s for s in sources if s["name"] == args.only]
        if not sources:
            raise SystemExit(f"Source inconnue : {args.only}")

    results = []
    for source in sources:
        print(f"[{source['name']}] {source['hf_id']}")
        results.append(explore_source(source))

    inventory = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "sources": results}
    REPORT_DIR.mkdir(exist_ok=True)
    suffix = f"_{args.only}" if args.only else ""
    (REPORT_DIR / f"data_inventory{suffix}.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / f"data_inventory{suffix}.md").write_text(
        to_markdown(inventory), encoding="utf-8")
    print(f"Rapport écrit dans {REPORT_DIR}")


if __name__ == "__main__":
    main()
