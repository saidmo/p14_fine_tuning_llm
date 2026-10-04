"""Profilage ciblé des sources, pour fonder les choix de constitution des datasets.

Questions mesurées :
  - MediQAl       : répartition par spécialité / type de question, part des cas cliniques
  - FrenchMedMCQA : confirmation que le miroir ne contient que des QCU
  - MedQuAD       : distribution des qtype, doublons, longueurs en tokens (tokenizer Qwen3)
  - UltraMedical  : label_type, sources (préfixe de prompt_id), conflits label / scores,
                    longueurs en tokens (échantillon)
  - Recouvrement  : questions communes entre splits et entre MediQAl / FrenchMedMCQA

Sorties : reports/profiling.json et reports/profiling.md

Usage (depuis la racine du projet) :
    python -m src.data.profile_sources
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Iterable

import numpy as np
from transformers import AutoTokenizer

from src.data.sources import ROOT, load_source

REPORT_DIR = ROOT / "reports"
TOKENIZER_ID = "Qwen/Qwen3-1.7B-Base"
SEED = 42
DPO_TOKEN_SAMPLE = 5000            # paires UltraMedical tokenisées (échantillon aléatoire)
TOKEN_THRESHOLDS = (512, 1024, 2048)


# ---------------------------------------------------------------- utilitaires
def normalize(text: str | None) -> str:
    """Minuscules, sans accents ni ponctuation, espaces réduits (clé de comparaison)."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def mcq_key(row: dict[str, Any]) -> str:
    """Clé d'un QCM = question + 5 options : la question seule est souvent trop générique."""
    parts = [row["question"]] + [row[f"answer_{letter}"] for letter in "abcde"]
    return normalize(" ".join(p or "" for p in parts))


def token_lengths(tokenizer, texts: Iterable[str | None]) -> list[int]:
    encoded = tokenizer([t or "" for t in texts], add_special_tokens=False)["input_ids"]
    return [len(ids) for ids in encoded]


def length_stats(lengths: list[int]) -> dict[str, int]:
    arr = np.asarray(lengths)
    stats = {f"p{p}": int(np.percentile(arr, p)) for p in (50, 90, 95, 99)}
    stats["max"] = int(arr.max())
    stats.update({f"nb > {t}": int((arr > t).sum()) for t in TOKEN_THRESHOLDS})
    return stats


def last_assistant(messages: list[dict[str, str]]) -> str:
    """chosen / rejected contiennent toute la conversation : on isole la réponse."""
    return next((m["content"] for m in reversed(messages) if m["role"] == "assistant"), "")


# ---------------------------------------------------------------- par source
def profile_mediqal() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cfg in ("mcqu", "mcqm", "oeq"):
        out[cfg] = {}
        for split, d in load_source("mediqal", cfg).items():
            out[cfg][split] = {
                "rows": len(d),
                "with_clinical_case": sum(1 for c in d["clinical_case"] if c),
                "question_type": dict(Counter(d["question_type"])),
                "medical_subject": dict(Counter(d["medical_subject"]).most_common()),
            }
    return out


def profile_frenchmedmcqa() -> dict[str, Any]:
    out = {}
    for split, d in load_source("frenchmedmcqa").items():
        out[split] = {
            "rows": len(d),
            "features": {k: str(d.features[k]) for k in ("correct_answers", "number_correct_answers")},
            "number_correct_answers": dict(Counter(d["number_correct_answers"])),
            "correct_answers": dict(Counter(d["correct_answers"])),
        }
    return out


def profile_medquad(tokenizer) -> dict[str, Any]:
    d = load_source("medquad")["train"]
    questions = [normalize(q) for q in d["Question"]]
    pairs = list(zip(questions, (normalize(a) for a in d["Answer"])))
    return {
        "rows": len(d),
        "qtype": dict(Counter(d["qtype"]).most_common()),
        "duplicate_questions": len(questions) - len(set(questions)),
        "duplicate_qa_pairs": len(pairs) - len(set(pairs)),
        "empty_answers": sum(1 for a in d["Answer"] if not (a or "").strip()),
        "tokens_question": length_stats(token_lengths(tokenizer, d["Question"])),
        "tokens_answer": length_stats(token_lengths(tokenizer, d["Answer"])),
    }


def score_verdict(meta: dict[str, Any]) -> str:
    chosen = (meta.get("chosen") or {}).get("score")
    rejected = (meta.get("rejected") or {}).get("score")
    if chosen is None or rejected is None:
        return "score_absent"
    if chosen > rejected:
        return "chosen > rejected"
    return "egalite" if chosen == rejected else "chosen < rejected"


def profile_ultramedical(tokenizer) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split, d in load_source("ultramedical_preference").items():
        verdicts: dict[str, Counter] = {}
        for label, meta in zip(d["label_type"], d["metadata"]):
            verdicts.setdefault(label, Counter())[score_verdict(meta)] += 1
        out[split] = {
            "rows": len(d),
            "label_type": dict(Counter(d["label_type"]).most_common()),
            "source": dict(Counter(pid.split(",")[0] for pid in d["prompt_id"]).most_common()),
            "verdict_scores_par_label_type": {k: dict(v) for k, v in verdicts.items()},
        }

    # Longueurs en tokens sur un échantillon du train (fixe max_length du DPO)
    train = load_source("ultramedical_preference")["train"]
    sample = train.shuffle(seed=SEED).select(range(min(DPO_TOKEN_SAMPLE, len(train))))
    out["train_tokens_echantillon"] = {
        "taille": len(sample),
        "prompt": length_stats(token_lengths(tokenizer, sample["prompt"])),
        "chosen": length_stats(token_lengths(tokenizer, (last_assistant(m) for m in sample["chosen"]))),
        "rejected": length_stats(token_lengths(tokenizer, (last_assistant(m) for m in sample["rejected"]))),
    }
    return out


def profile_overlap() -> dict[str, Any]:
    keys: dict[str, list[str]] = {}
    for cfg in ("mcqu", "mcqm"):
        for split, d in load_source("mediqal", cfg).items():
            keys[f"mediqal/{cfg}/{split}"] = [mcq_key(r) for r in d]
    for split, d in load_source("frenchmedmcqa").items():
        keys[f"frenchmedmcqa/{split}"] = [mcq_key(r) for r in d]

    sets = {name: set(k) for name, k in keys.items()}
    return {
        "doublons_internes": {name: len(k) - len(sets[name]) for name, k in keys.items()},
        "recouvrements": {f"{a} & {b}": n for a, b in combinations(sets, 2)
                          if (n := len(sets[a] & sets[b]))},
    }


# ---------------------------------------------------------------- rapport
def main() -> None:
    print(f"Tokenizer : {TOKENIZER_ID}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    sections = {}
    for name, func in [("mediqal", profile_mediqal),
                       ("frenchmedmcqa", profile_frenchmedmcqa),
                       ("medquad", lambda: profile_medquad(tokenizer)),
                       ("ultramedical_preference", lambda: profile_ultramedical(tokenizer)),
                       ("recouvrement_mcq", profile_overlap)]:
        print(f"[{name}] profilage…")
        sections[name] = func()

    report = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "tokenizer": TOKENIZER_ID, "seed": SEED, **sections}

    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "profiling.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [f"# Profilage des sources — {report['generated_at']}",
          "", f"Tokenizer : `{TOKENIZER_ID}` — seed : {SEED}"]
    for name, content in sections.items():
        md += ["", f"## {name}", "", "```json",
               json.dumps(content, ensure_ascii=False, indent=2), "```"]
    (REPORT_DIR / "profiling.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Rapport écrit dans {REPORT_DIR}")


if __name__ == "__main__":
    main()
