"""Formatage final : jeux anonymisés -> data/processed/ (format conversationnel TRL).

  - SFT  : messages = [system, user, assistant] + métadonnées
  - DPO  : prompt = [system, user], chosen = [assistant], rejected = [assistant] + métadonnées
Le chat template de Qwen3 n'est PAS stocké dans les données : il est appliqué par le
tokenizer à l'entraînement comme à l'inférence (vLLM), ce qui garantit le même gabarit.

Contrôles (fermés par défaut) :
  - refus si l'audit des noms (reports/audit_noms.json) n'est pas au statut « valide » ;
  - longueurs mesurées avec le VRAI chat template, consigne système comprise ; arrêt si un
    exemple d'entraînement ou de validation dépasse les limites d'ENTRAÎNEMENT :
      SFT  : max_seq_length (la sélection a réservé marge_template sur le contenu brut)
      DPO  : prompt <= max_prompt_length + marge_template (même logique : 512 + 96 = 608) ;
             total <= max_length. Ces valeurs sont celles à passer à TRL (DPOConfig).
    Les jeux de test ne bloquent pas : ils servent à l'inférence (contexte de Qwen3 plus long).

Champs de travail non publiés : dedup_key (empreinte), n_tokens, meta (sauf qtype).

Sorties : data/processed/<jeu>.jsonl et reports/format.json

Usage (depuis la racine du projet) :
    python -m src.data.build_dataset
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import numpy as np
import yaml
from transformers import AutoTokenizer

from src.data.sources import ROOT

ANON_DIR = ROOT / "data" / "interim" / "anonymise"
OUTPUT_DIR = ROOT / "data" / "processed"
CONFIG_PATH = ROOT / "configs" / "format.yaml"
SELECTION_CONFIG_PATH = ROOT / "configs" / "selection.yaml"
AUDIT_REPORT_PATH = ROOT / "reports" / "audit_noms.json"
REPORT_PATH = ROOT / "reports" / "format.json"

SFT_SETS = ["sft_train", "sft_validation", "sft_test", "eval_clinique"]
DPO_SETS = ["dpo_train", "dpo_validation", "dpo_test"]
TRAINING_SETS = {"sft_train", "sft_validation", "dpo_train", "dpo_validation"}   # longueurs bloquantes
ENTITY_TYPES = ("PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "URL")


def read_jsonl(path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_audit() -> None:
    if not AUDIT_REPORT_PATH.exists():
        raise SystemExit("Audit des noms absent : lancer src.data.audit_names avant le formatage.")
    status = json.loads(AUDIT_REPORT_PATH.read_text(encoding="utf-8"))["statut"]
    if status != "valide":
        raise SystemExit(f"Audit des noms au statut « {status} » : formatage refusé (fermé par défaut).")


def load_tokenizer(cfg: dict[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    source = cfg["tokenizer"]
    if not tokenizer.chat_template:
        tokenizer.chat_template = AutoTokenizer.from_pretrained(cfg["chat_template_source"]).chat_template
        source = cfg["chat_template_source"]
    return tokenizer, source


def count_tokens(tokenizer, messages: list[dict[str, str]], cfg: dict[str, Any],
                 generation_prompt: bool = False) -> int:
    out = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation_prompt,
                                        enable_thinking=cfg["enable_thinking"])
    # Selon la version de transformers : liste d'identifiants, ou BatchEncoding (hérite de
    # UserDict, PAS de dict) contenant input_ids et attention_mask — len() compterait alors
    # les 2 clés au lieu des tokens.
    if hasattr(out, "keys") and "input_ids" in out.keys():
        out = out["input_ids"]
    if out and isinstance(out[0], list):          # forme « lot » d'une seule conversation
        out = out[0]
    if len(out) < 10:
        raise SystemExit(f"Comptage de tokens suspect ({len(out)}) : vérifier la sortie de apply_chat_template.")
    return len(out)


def masked_entities(record: dict[str, Any]) -> dict[str, int]:
    """Comptage à clés fixes : un schéma stable pour le format Hugging Face Datasets."""
    counts = record.get("anonymisation") or {}
    return {entity: int(counts.get(entity, 0)) for entity in ENTITY_TYPES}


def format_sft(record: dict[str, Any], system: str) -> dict[str, Any]:
    return {
        "id": record["id"],
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": record["instruction"]},
                     {"role": "assistant", "content": record["response"]}],
        "source": record["source"], "config": record["config"],
        "source_revision": record["source_revision"], "langue": record["langue"],
        "task": record["task"], "medical_subject": record.get("medical_subject"),
        "question_type": record.get("question_type"), "qtype": (record.get("meta") or {}).get("qtype"),
        "has_clinical_case": bool(record.get("has_clinical_case")),
        "niveau_confiance": record["niveau_confiance"],
        "bloc": record.get("bloc"), "complement_de": record.get("complement_de"),
        # Schéma de métadonnées cliniques demandé par la mission : non renseigné par ces sources
        "symptomes": None, "antecedents": None, "constantes": None,
        "transformations": record.get("transformations", []),
        "entites_masquees": masked_entities(record),
    }


def format_dpo(record: dict[str, Any], system: str) -> dict[str, Any]:
    reduction = record.get("reduction_chatdoctor")
    return {
        "id": record["id"],
        "prompt": [{"role": "system", "content": system}, {"role": "user", "content": record["prompt"]}],
        "chosen": [{"role": "assistant", "content": record["chosen"]}],
        "rejected": [{"role": "assistant", "content": record["rejected"]}],
        "origine": record.get("origine"), "label_type": record.get("label_type"),
        "score_chosen": None if record.get("score_chosen") is None else float(record["score_chosen"]),
        "score_rejected": None if record.get("score_rejected") is None else float(record["score_rejected"]),
        "source_revision": record["source_revision"], "langue": record["langue"],
        "niveau_confiance": record["niveau_confiance"],
        "transformations": record.get("transformations", []),
        "entites_masquees": masked_entities(record),
        # chaîne JSON : ses clés varient d'une paire à l'autre (règles appliquées)
        "reduction_chatdoctor": None if reduction is None else json.dumps(reduction, ensure_ascii=False),
    }


def length_stats(lengths: list[int]) -> dict[str, int]:
    if not lengths:
        return {}
    arr = np.asarray(lengths)
    return {"p50": int(np.percentile(arr, 50)), "p95": int(np.percentile(arr, 95)), "max": int(arr.max())}


def main() -> None:
    check_audit()
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    selection = yaml.safe_load(SELECTION_CONFIG_PATH.read_text(encoding="utf-8"))
    max_sft = selection["sft"]["max_seq_length"]
    # Limite du prompt à l'entraînement = limite du contenu brut + marge réservée au template,
    # comme pour le SFT (la sélection filtrait le prompt brut à max_prompt_length)
    max_prompt = selection["dpo"]["max_prompt_length"] + selection["dpo"]["marge_template"]
    max_dpo = selection["dpo"]["max_length"]
    system = cfg["system_prompt"].strip()

    tokenizer, template_source = load_tokenizer(cfg)
    print(f"Chat template : {template_source}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              "system_prompt": system, "chat_template": template_source,
                              "enable_thinking": cfg["enable_thinking"],
                              "limites_entrainement": {"sft_max_seq_length": max_sft,
                                                       "dpo_max_prompt_length": max_prompt,
                                                       "dpo_max_length": max_dpo},
                              "jeux": {}}
    blocking: list[str] = []

    for name in SFT_SETS + DPO_SETS:
        is_sft = name in SFT_SETS
        records = [(format_sft if is_sft else format_dpo)(r, system) for r in read_jsonl(ANON_DIR / f"{name}.jsonl")]
        too_long, lengths, prompt_lengths = [], [], []
        for r in records:
            if is_sft:
                n = count_tokens(tokenizer, r["messages"], cfg)
                lengths.append(n)
                if n > max_sft:
                    too_long.append(r["id"])
            else:
                n_prompt = count_tokens(tokenizer, r["prompt"], cfg, generation_prompt=True)
                prompt_lengths.append(n_prompt)
                n_full = max(count_tokens(tokenizer, r["prompt"] + r["chosen"], cfg),
                             count_tokens(tokenizer, r["prompt"] + r["rejected"], cfg))
                lengths.append(n_full)
                if n_prompt > max_prompt or n_full > max_dpo:
                    too_long.append(r["id"])

        with (OUTPUT_DIR / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        key = "task" if is_sft else "origine"
        report["jeux"][name] = {"total": len(records), "langue": dict(Counter(r["langue"] for r in records)),
                                key: dict(Counter(str(r[key]) for r in records).most_common()),
                                "tokens_avec_template": length_stats(lengths),
                                **({"tokens_prompt_avec_template": length_stats(prompt_lengths)} if not is_sft else {}),
                                "depassements": len(too_long), "ids_depassements": too_long[:20]}
        prompt_info = "" if is_sft else f" — prompt {length_stats(prompt_lengths)}"
        print(f"  {name:<15} {len(records):>6} — tokens {length_stats(lengths)}{prompt_info} — "
              f"dépassements : {len(too_long)}")
        if too_long and name in TRAINING_SETS:
            blocking.append(f"{name} : {len(too_long)}")

    report["statut"] = "echec" if blocking else "succes"
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Rapport : {REPORT_PATH}")
    if blocking:
        raise SystemExit("Exemples d'entraînement trop longs avec le chat template (" + ", ".join(blocking)
                         + ") : augmenter marge_template dans selection.yaml et relancer la chaîne.")
    print(f"Limites à passer à TRL : SFT max_seq_length={max_sft} ; "
          f"DPO max_prompt_length={max_prompt}, max_length={max_dpo}")


if __name__ == "__main__":
    main()
