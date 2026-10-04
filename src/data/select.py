"""Sélection, dédoublonnage et découpage des datasets SFT et DPO.

Règles appliquées (paramètres dans configs/selection.yaml) :
  1. Priorité test > validation > train : les jeux de test restent intacts ;
     tout exemple de validation présent dans un test est retiré ; tout exemple
     de train présent dans un test ou une validation est retiré ; doublons
     internes supprimés (première occurrence conservée).
  2. MedQuAD (sans split officiel) : split déterministe par hash de la clé,
     donc les doublons tombent toujours dans le même split.
  3. Filtres par bloc (spécialités, qtypes, longueur en tokens), puis tirage STABLE :
     chaque exemple reçoit un rang fixe sha256(seed, bloc, identifiant) et on prend les
     premiers (priorité éventuelle d'abord). Retirer un exemple ne change pas le rang des
     autres : une exclusion ne fait entrer que ses remplaçants. Contrôle des quotas.
  4. DPO : label_type retenus, scores cohérents (chosen > rejected), origines
     ciblées, longueurs max.
  5. Exclusions d'anonymisation (configs/exclusions_anonymisation.yaml, produit par
     audit_names) : écartées de TOUS les jeux, tests compris (la protection des données
     prime sur la comparabilité du test), avant toute autre règle ; les quotas sont
     complétés depuis les réserves.

Sorties : data/interim/selection/*.jsonl et reports/selection.json

Usage (depuis la racine du projet) :
    python -m src.data.select
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import yaml
from transformers import AutoTokenizer

from src.data.sources import ROOT

CONFIG_PATH = ROOT / "configs" / "selection.yaml"
INTERIM_DIR = ROOT / "data" / "interim"
OUTPUT_DIR = INTERIM_DIR / "selection"
REPORT_PATH = ROOT / "reports" / "selection.json"
EXCLUSIONS_PATH = ROOT / "configs" / "exclusions_anonymisation.yaml"

Record = dict[str, Any]

PRIORITIES: dict[str, Callable[[Record], int]] = {
    # 0 = prioritaire : cas clinique ou question de raisonnement
    "clinique": lambda r: 0 if r["has_clinical_case"] or r["question_type"] == "Reasoning" else 1,
}


# ---------------------------------------------------------------- utilitaires
def read_jsonl(path: Path) -> Iterator[Record]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[Record]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stable_bucket(key: str) -> int:
    """Seau 0-99 stable (sha256, contrairement à hash() qui varie d'une exécution à l'autre)."""
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 100


def token_counts(tokenizer, texts: Iterable[str]) -> list[int]:
    texts = list(texts)
    if not texts:
        return []
    return [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]


def stable_rank(record_id: str, salt: str) -> str:
    """Rang pseudo-aléatoire FIXE d'un exemple : ne dépend que de son identifiant et du sel
    (seed + bloc). Contrairement à rng.shuffle(), retirer un exemple du pool ne modifie pas
    le rang des autres — sans cela, 9 exclusions renouvelaient des centaines d'exemples."""
    return hashlib.sha256(f"{salt}:{record_id}".encode("utf-8")).hexdigest()


def sample(candidates: list[Record], n: int, salt: str,
           priority: Callable[[Record], int] | None = None) -> tuple[list[Record], list[Record]]:
    """Tirage stable : tri par (priorité, rang fixe). Retourne (retenus, réserve ordonnée)."""
    pool = sorted(candidates, key=lambda r: ((priority(r) if priority else 0), stable_rank(r["id"], salt)))
    return pool[:n], pool[n:]


def composition(records: list[Record], fields: Iterable[str]) -> dict[str, dict[str, int]]:
    return {f: dict(Counter(str(r.get(f)) for r in records).most_common()) for f in fields}


class Removals:
    """Compte les exemples écartés, par jeu de données et par motif."""

    def __init__(self) -> None:
        self.counts: dict[str, Counter] = defaultdict(Counter)

    def add(self, dataset: str, reason: str) -> None:
        self.counts[dataset][reason] += 1

    def to_dict(self) -> dict[str, dict[str, int]]:
        return {k: dict(v) for k, v in self.counts.items()}


def load_exclusions() -> set[str]:
    """Identifiants retirés par précaution lors du contrôle exhaustif des noms."""
    if not EXCLUSIONS_PATH.exists():
        return set()
    data = yaml.safe_load(EXCLUSIONS_PATH.read_text(encoding="utf-8")) or {}
    return {e["id"] for e in data.get("exemples_exclus") or []}


def drop_excluded(records: list[Record], excluded: set[str], removals: Removals,
                  dataset: str) -> list[Record]:
    kept = [r for r in records if r["id"] not in excluded]
    for _ in range(len(records) - len(kept)):
        removals.add(dataset, "exclusion_anonymisation")
    return kept


def deduplicate(records: list[Record], forbidden: set[str], removals: Removals,
                dataset: str) -> list[Record]:
    kept, seen = [], set()
    for r in records:
        key = r["dedup_key"]
        if key in forbidden:
            removals.add(dataset, "present_dans_jeu_prioritaire")
        elif key in seen:
            removals.add(dataset, "doublon_interne")
        else:
            seen.add(key)
            kept.append(r)
    return kept


# ---------------------------------------------------------------- SFT
def bloc_exclusion(r: Record, bloc: dict[str, Any]) -> str | None:
    """Motif d'exclusion d'un exemple pour un bloc, ou None s'il est éligible."""
    if "configs" in bloc and r["config"] not in bloc["configs"]:
        return "config_hors_bloc"
    if r.get("medical_subject") in bloc.get("sujets_exclus", []):
        return "sujet_exclu"
    if "qtypes" in bloc and r["meta"].get("qtype") not in bloc["qtypes"]:
        return "qtype_exclu"
    return None


def eligible(records: list[Record], bloc: dict[str, Any], tokenizer, max_content: int,
             removals: Removals, dataset: str) -> list[Record]:
    kept = []
    for r in records:
        reason = bloc_exclusion(r, bloc)
        if reason:
            removals.add(dataset, reason)
        else:
            kept.append(r)

    result = []
    lengths = token_counts(tokenizer, (f"{r['instruction']}\n{r['response']}" for r in kept))
    for r, n in zip(kept, lengths):
        if n > max_content:
            removals.add(dataset, "trop_long")
        else:
            result.append({**r, "n_tokens": n})
    return result


def compose_sft(split: str, candidates: list[Record], sft_cfg: dict[str, Any], tokenizer,
                rng: random.Random, removals: Removals, seed: Any) -> tuple[list[Record], dict[str, Any]]:
    dataset = f"sft_{split}"
    max_content = sft_cfg["max_seq_length"] - sft_cfg["marge_template"]
    selected: dict[str, list[Record]] = {}
    reserves: dict[str, list[Record]] = {}
    report: dict[str, Any] = {}

    # 1er passage : chaque bloc tire dans ses propres candidats
    for name, bloc in sft_cfg["blocs"].items():
        if "fichier" in bloc:
            path = ROOT / bloc["fichier"]
            pool = [r for r in read_jsonl(path) if r["split"] == split] if path.exists() else []
        else:
            pool = [r for r in candidates if r["source"] == bloc["source"]]
        pool = eligible(pool, bloc, tokenizer, max_content, removals, dataset)
        chosen, rest = sample(pool, bloc[split], f"{seed}:{split}:{name}", PRIORITIES.get(bloc.get("priorite")))
        selected[name] = [{**r, "bloc": name} for r in chosen]
        reserves[name] = rest
        report[name] = {"quota": bloc[split], "eligibles": len(pool), "retenus": len(chosen)}

    # 2e passage : les blocs déficitaires sont complétés depuis la réserve désignée
    for name, bloc in sft_cfg["blocs"].items():
        missing = bloc[split] - len(selected[name])
        donor = bloc.get("complement")
        if missing > 0 and donor:
            extra, reserves[donor] = reserves[donor][:missing], reserves[donor][missing:]
            selected[name] += [{**r, "bloc": donor, "complement_de": name} for r in extra]
            report[name]["complement"] = {"depuis": donor, "nombre": len(extra)}

    for name, bloc in sft_cfg["blocs"].items():
        if len(selected[name]) != bloc[split]:
            raise SystemExit(f"[{dataset}] bloc '{name}' : {len(selected[name])} exemples "
                             f"pour un quota de {bloc[split]}")

    records = [r for name in sft_cfg["blocs"] for r in selected[name]]
    rng.shuffle(records)
    return records, report


def build_sft(cfg: dict[str, Any], tokenizer, removals: Removals, excluded: set[str]) -> dict[str, Any]:
    sft_cfg = cfg["sft"]
    quota_total = sum(b["train"] for b in sft_cfg["blocs"].values())
    if quota_total != sft_cfg["total_train"]:
        raise SystemExit(f"Somme des quotas train = {quota_total}, attendu {sft_cfg['total_train']}")

    # Exclusions d'anonymisation en premier : jamais réintégrées par une autre règle
    mediqal = drop_excluded(list(read_jsonl(INTERIM_DIR / "mediqal.jsonl")), excluded, removals, "sources_sft")
    frenchmedmcqa = drop_excluded(list(read_jsonl(INTERIM_DIR / "frenchmedmcqa.jsonl")), excluded,
                                  removals, "sources_sft")
    medquad = drop_excluded(list(read_jsonl(INTERIM_DIR / "medquad.jsonl")), excluded, removals, "sources_sft")

    # MedQuAD : split déterministe par hash de la clé de dédoublonnage
    split_cfg = sft_cfg["split_medquad"]
    for r in medquad:
        bucket = stable_bucket(r["dedup_key"])
        r["split"] = ("test" if bucket < split_cfg["test_pct"] else
                      "validation" if bucket < split_cfg["test_pct"] + split_cfg["validation_pct"]
                      else "train")
        r["transformations"].append("split_par_hash")

    eval_cfg = sft_cfg["eval_clinique"]
    eval_clinique = [r for r in mediqal if r["config"] == eval_cfg["config"]
                     and r["medical_subject"] in eval_cfg["sujets"]]

    pool = [r for r in mediqal if r["config"] != eval_cfg["config"]] + frenchmedmcqa + medquad
    tests = [r for r in pool if r["split"] == "test"]                    # intacts
    test_keys = {r["dedup_key"] for r in tests} | \
                {r["dedup_key"] for r in mediqal if r["config"] == eval_cfg["config"]}
    validation = deduplicate([r for r in pool if r["split"] == "validation"],
                             test_keys, removals, "sft_validation")
    val_keys = {r["dedup_key"] for r in validation}
    train = deduplicate([r for r in pool if r["split"] == "train"],
                        test_keys | val_keys, removals, "sft_train")

    seed = cfg["seed"]
    train_sel, train_report = compose_sft("train", train, sft_cfg, tokenizer,
                                          random.Random(f"{seed}:train"), removals, seed)
    val_sel, val_report = compose_sft("validation", validation, sft_cfg, tokenizer,
                                      random.Random(f"{seed}:validation"), removals, seed)

    if len(train_sel) != sft_cfg["total_train"]:
        raise SystemExit(f"sft_train : {len(train_sel)} exemples, attendu {sft_cfg['total_train']}")

    return {"datasets": {"sft_train": train_sel, "sft_validation": val_sel,
                         "sft_test": tests, "eval_clinique": eval_clinique},
            "quotas": {"sft_train": train_report, "sft_validation": val_report}}


# ---------------------------------------------------------------- DPO
def select_dpo(candidates: list[Record], quotas: dict[str, int], dpo_cfg: dict[str, Any],
               tokenizer, rng: random.Random, removals: Removals,
               dataset: str) -> tuple[list[Record], dict[str, Any]]:
    max_length = dpo_cfg["max_length"] - dpo_cfg["marge_template"]
    selected, report = [], {}
    for origine, quota in quotas.items():
        pool = sorted((r for r in candidates if r["origine"] == origine),
                      key=lambda r: stable_rank(r["id"], f"{dataset}:{origine}:{dpo_cfg['seed']}"))
        kept = []
        for r in pool:                      # on ne tokenise que jusqu'à remplir le quota
            if len(kept) == quota:
                break
            p, c, j = token_counts(tokenizer, [r["prompt"], r["chosen"], r["rejected"]])
            if p > dpo_cfg["max_prompt_length"] or p + max(c, j) > max_length:
                removals.add(dataset, "trop_long")
                continue
            kept.append({**r, "n_tokens": {"prompt": p, "chosen": c, "rejected": j}})
        if len(kept) < quota:
            raise SystemExit(f"[{dataset}] origine '{origine}' : {len(kept)} paires pour un quota de {quota}")
        selected += kept
        report[origine] = {"quota": quota, "candidats": len(pool), "retenus": len(kept)}
    rng.shuffle(selected)
    return selected, report


def build_dpo(cfg: dict[str, Any], tokenizer, removals: Removals, excluded: set[str]) -> dict[str, Any]:
    dpo_cfg = {**cfg["dpo"], "seed": cfg["seed"]}
    allowed = set(dpo_cfg["quotas_train"]) | set(dpo_cfg["quotas_validation"])
    by_split: dict[str, list[Record]] = defaultdict(list)

    # Lecture en flux : le fichier UltraMedical est volumineux
    for r in read_jsonl(INTERIM_DIR / "ultramedical_preference.jsonl"):
        split = r["split"]
        if r["id"] in excluded:              # avant tout, test compris
            removals.add(f"dpo_{split}", "exclusion_anonymisation")
            continue
        if split == "test":                 # jeu de test conservé intact
            by_split["test"].append(r)
            continue
        dataset = f"dpo_{split}"
        if r["label_type"] not in dpo_cfg["label_types"]:
            removals.add(dataset, "label_type_exclu")
        elif (r["score_chosen"] is None or r["score_rejected"] is None
              or r["score_chosen"] <= r["score_rejected"]):
            removals.add(dataset, "scores_incoherents")
        elif r["origine"] not in allowed:
            removals.add(dataset, "origine_exclue")
        else:
            by_split[split].append(r)

    test_keys = {r["dedup_key"] for r in by_split["test"]}
    validation = deduplicate(by_split["validation"], test_keys, removals, "dpo_validation")
    train = deduplicate(by_split["train"], test_keys | {r["dedup_key"] for r in validation},
                        removals, "dpo_train")

    seed = cfg["seed"]
    train_sel, train_report = select_dpo(train, dpo_cfg["quotas_train"], dpo_cfg, tokenizer,
                                         random.Random(f"{seed}:dpo_train"), removals, "dpo_train")
    val_sel, val_report = select_dpo(validation, dpo_cfg["quotas_validation"], dpo_cfg, tokenizer,
                                     random.Random(f"{seed}:dpo_validation"), removals,
                                     "dpo_validation")
    return {"datasets": {"dpo_train": train_sel, "dpo_validation": val_sel,
                         "dpo_test": by_split["test"]},
            "quotas": {"dpo_train": train_report, "dpo_validation": val_report}}


# ---------------------------------------------------------------- exécution
def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    print(f"Tokenizer : {cfg['tokenizer']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    removals = Removals()
    excluded = load_exclusions()
    print(f"Exclusions d'anonymisation : {len(excluded)}")

    try:
        print("[sft] sélection…")
        sft = build_sft(cfg, tokenizer, removals, excluded)
        print("[dpo] sélection…")
        dpo = build_dpo(cfg, tokenizer, removals, excluded)
    except SystemExit as exc:
        REPORT_PATH.parent.mkdir(exist_ok=True)
        REPORT_PATH.write_text(json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "statut": "echec", "erreur": str(exc), "configuration": cfg,
             "exclusions": removals.to_dict()}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Rapport d'échec : {REPORT_PATH}")
        raise

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {**sft["datasets"], **dpo["datasets"]}
    sft_fields = ("source", "bloc", "task", "langue", "has_clinical_case", "medical_subject")
    dpo_fields = ("origine", "label_type")
    contenu = {}
    for name, records in datasets.items():
        write_jsonl(OUTPUT_DIR / f"{name}.jsonl", records)
        contenu[name] = {"total": len(records),
                         **composition(records, dpo_fields if name.startswith("dpo") else sft_fields)}
        print(f"  {name:<16} {len(records):>6}")

    report = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "statut": "succes", "exclusions_anonymisation_appliquees": len(excluded),
              "configuration": cfg, "contenu": contenu,
              "quotas": {**sft["quotas"], **dpo["quotas"]},
              "exclusions": removals.to_dict()}
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Rapport : {REPORT_PATH}")


if __name__ == "__main__":
    main()
