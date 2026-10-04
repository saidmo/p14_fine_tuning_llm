"""Normalisation des sources vers le schéma commun (data/interim/).

Étape de mise en forme uniquement : aucune sélection, aucun échantillonnage,
aucun dédoublonnage. Chaque ligne source donne un enregistrement normalisé
ou un rejet motivé. Filtres et composition des datasets : étape suivante.

Sorties :
    data/interim/<source>.jsonl          enregistrements normalisés (tous splits)
    data/interim/<source>.rejets.jsonl   lignes écartées, avec la raison
    reports/normalization.json           volumes, rejets et transformations par source

Usage (depuis la racine du projet) :
    python -m src.data.normalize
    python -m src.data.normalize --only mediqal
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterator

from src.data.schema import (QUESTION_TYPE_MAP, SUBJECT_MAP, DPORecord, SFTRecord,
                             clean_text, dedup_key, normalize_label)
from src.data.sources import ROOT, load_registry, load_source

INTERIM_DIR = ROOT / "data" / "interim"
REPORT_PATH = ROOT / "reports" / "normalization.json"
LETTERS = "ABCDE"
REQUIRED_FIELDS = ("hf_id", "langue", "revision", "niveau_confiance")


CONSIGNES_QCM = {
    "mcq_single": "Une seule réponse est attendue. Indiquez la lettre et le texte de la réponse choisie.",
    "mcq_multi": "Une ou plusieurs réponses sont attendues. Indiquez les lettres et le texte "
                 "de chacune des réponses choisies.",
}


@dataclass
class Reject:
    source: str
    split: str
    source_id: str
    raison: str


Item = SFTRecord | DPORecord | Reject


# ---------------------------------------------------------------- briques communes
def _common(src: dict[str, Any], config: str, split: str, source_id: str) -> dict[str, Any]:
    return {"id": f"{src['name']}:{config}:{split}:{source_id}", "source": src["name"],
            "config": config, "source_id": source_id, "source_revision": src["revision"],
            "langue": src["langue"], "split": split, "niveau_confiance": src["niveau_confiance"]}


def _safe(build: Callable[[], Item], src: dict[str, Any], split: str, source_id: str) -> Item:
    """Construit l'enregistrement ; toute incohérence devient un rejet motivé."""
    try:
        return build()
    except ValueError as exc:
        return Reject(src["name"], split, source_id, str(exc))


def format_mcq(question: str, options: dict[str, str], case: str, task: str) -> str:
    parts = [f"Cas clinique :\n{case}"] if case else []
    parts += [f"Question : {question}",
              "\n".join(f"{letter}. {text}" for letter, text in options.items()),
              CONSIGNES_QCM[task]]
    return "\n\n".join(parts)


def format_mcq_answer(correct: list[str], options: dict[str, str], task: str) -> str:
    if task == "mcq_single":
        return f"Réponse : {correct[0]}. {options[correct[0]]}"
    lines = [f"Réponses : {', '.join(correct)}"]
    lines += [f"- {letter}. {options[letter]}" for letter in correct]
    return "\n".join(lines)


def mcq_record(common: dict[str, Any], task: str, question: str | None,
               raw_options: dict[str, str | None], correct: list[str], case: str | None,
               subject: str | None, qtype: str | None, meta: dict[str, Any] | None = None) -> SFTRecord:
    log: list[str] = []
    question = clean_text(question, log)
    case = clean_text(case, log)
    options = {letter: text for letter in LETTERS
               if (text := clean_text(raw_options.get(letter), log))}

    if len(options) < 2:
        raise ValueError("moins de deux options non vides")
    if not correct:
        raise ValueError("aucune bonne réponse")
    if any(letter not in options for letter in correct):
        raise ValueError("bonne réponse absente des options")
    if task == "mcq_single" and len(correct) != 1:
        raise ValueError("QCU avec plusieurs bonnes réponses")

    return SFTRecord(**common, task=task,
                     instruction=format_mcq(question, options, case, task),
                     response=format_mcq_answer(correct, options, task),
                     dedup_key=dedup_key(question, *options.values()),
                     medical_subject=normalize_label(subject, SUBJECT_MAP, log),
                     question_type=normalize_label(qtype, QUESTION_TYPE_MAP, log),
                     has_clinical_case=bool(case), transformations=log, meta=meta or {})


def open_qa_record(common: dict[str, Any], question: str | None, answer: str | None,
                   case: str | None = None, subject: str | None = None,
                   qtype: str | None = None, meta: dict[str, Any] | None = None) -> SFTRecord:
    log: list[str] = []
    question = clean_text(question, log)
    answer = clean_text(answer, log)
    case = clean_text(case, log)
    instruction = f"Cas clinique :\n{case}\n\nQuestion : {question}" if case else question
    return SFTRecord(**common, task="open_qa", instruction=instruction, response=answer,
                     # le cas fait partie de la clé : « Quel diagnostic ? » seul est trop générique
                     dedup_key=dedup_key(case, question),
                     medical_subject=normalize_label(subject, SUBJECT_MAP, log),
                     question_type=normalize_label(qtype, QUESTION_TYPE_MAP, log),
                     has_clinical_case=bool(case), transformations=log, meta=meta or {})


def parse_letters(value: str | None) -> list[str]:
    """'C,D' -> ['C', 'D'] (dédoublonné, trié)."""
    return sorted({x.strip().upper() for x in (value or "").split(",") if x.strip()})


def last_assistant(messages: list[dict[str, str]]) -> str:
    return next((m["content"] for m in reversed(messages) if m["role"] == "assistant"), "")


# ---------------------------------------------------------------- loaders
def load_mediqal(src: dict[str, Any]) -> Iterator[Item]:
    for config, task in (("mcqu", "mcq_single"), ("mcqm", "mcq_multi")):
        for split, d in load_source(src["name"], config).items():
            for row in d:
                sid = str(row["id"])
                yield _safe(lambda: mcq_record(
                    _common(src, config, split, sid), task, row["question"],
                    {letter: row[f"answer_{letter.lower()}"] for letter in LETTERS},
                    parse_letters(row["correct_answers"]), row["clinical_case"],
                    row["medical_subject"], row["question_type"]), src, split, sid)

    for split, d in load_source(src["name"], "oeq").items():
        for row in d:
            sid = str(row["id"])
            yield _safe(lambda: open_qa_record(
                _common(src, "oeq", split, sid), row["question"], row["answer"],
                row["clinical_case"], row["medical_subject"], row["question_type"],
                meta={"cc_question_number": row["cc_question_number"]}), src, split, sid)


def load_frenchmedmcqa(src: dict[str, Any]) -> Iterator[Item]:
    for split, d in load_source(src["name"]).items():
        for row in d:
            sid = str(row["id"])
            # Miroir limité aux QCU : correct_answers est l'index (0-4) de la bonne option
            yield _safe(lambda: mcq_record(
                _common(src, "default", split, sid), "mcq_single", row["question"],
                {letter: row[f"answer_{letter.lower()}"] for letter in LETTERS},
                [LETTERS[row["correct_answers"]]], None,
                "Pharmacy", None,                    # sujet : examens de pharmacie (card)
                meta={"sujet_depuis": "dataset card"}), src, split, sid)


def load_medquad(src: dict[str, Any]) -> Iterator[Item]:
    d = load_source(src["name"])["train"]
    for index, row in enumerate(d):
        sid = str(index)                             # pas d'identifiant : index à révision figée
        yield _safe(lambda: open_qa_record(
            _common(src, "default", "unsplit", sid), row["Question"], row["Answer"],
            meta={"qtype": row["qtype"]}), src, "unsplit", sid)


def load_ultramedical(src: dict[str, Any]) -> Iterator[Item]:
    for split, d in load_source(src["name"]).items():
        for row in d:
            sid = row["prompt_id"]

            def build() -> DPORecord:
                log: list[str] = []
                meta = row["metadata"] or {}
                chosen_meta, rejected_meta = meta.get("chosen") or {}, meta.get("rejected") or {}
                prompt = clean_text(row["prompt"], log)
                return DPORecord(
                    **_common(src, "default", split, sid),
                    dedup_key=dedup_key(prompt), prompt=prompt,
                    chosen=clean_text(last_assistant(row["chosen"]), log),
                    rejected=clean_text(last_assistant(row["rejected"]), log),
                    label_type=row["label_type"], origine=sid.split(",")[0],
                    score_chosen=chosen_meta.get("score"),
                    score_rejected=rejected_meta.get("score"),
                    transformations=log,
                    meta={"model_chosen": chosen_meta.get("model"),
                          "model_rejected": rejected_meta.get("model"),
                          "golden_answer": meta.get("golden_answer"),
                          "n_messages": len(row["chosen"])})

            yield _safe(build, src, split, sid)


LOADERS: dict[str, Callable[[dict[str, Any]], Iterator[Item]]] = {
    "mediqal": load_mediqal,
    "frenchmedmcqa": load_frenchmedmcqa,
    "medquad": load_medquad,
    "ultramedical_preference": load_ultramedical,
}


# ---------------------------------------------------------------- exécution
def run(name: str, src: dict[str, Any]) -> dict[str, Any]:
    missing = [f for f in REQUIRED_FIELDS if not src.get(f)]
    if missing:
        raise SystemExit(f"[{name}] champs manquants dans sources.yaml : {missing}")

    stats = {"ecrits": Counter(), "taches": Counter(), "rejets": Counter(),
             "transformations": Counter()}
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    with (INTERIM_DIR / f"{name}.jsonl").open("w", encoding="utf-8") as out, \
         (INTERIM_DIR / f"{name}.rejets.jsonl").open("w", encoding="utf-8") as rejects:
        for item in LOADERS[name](src):
            if isinstance(item, Reject):
                stats["rejets"][item.raison] += 1
                rejects.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
                continue
            record = item.to_dict()
            stats["ecrits"][record["split"]] += 1
            stats["taches"][record.get("task", "dpo")] += 1
            stats["transformations"].update(record["transformations"])
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "revision": src["revision"], **{k: dict(v) for k, v in stats.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=sorted(LOADERS), help="normaliser une seule source")
    args = parser.parse_args()

    registry = load_registry()
    names = [args.only] if args.only else list(LOADERS)

    report = (json.loads(REPORT_PATH.read_text(encoding="utf-8"))
              if REPORT_PATH.exists() else {"sources": {}})
    for name in names:
        print(f"[{name}] normalisation…")
        report["sources"][name] = run(name, {"name": name, **registry[name]})
        print(f"  écrits : {report['sources'][name]['ecrits']} — "
              f"rejets : {sum(report['sources'][name]['rejets'].values())}")

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Rapport : {REPORT_PATH}")


if __name__ == "__main__":
    main()
