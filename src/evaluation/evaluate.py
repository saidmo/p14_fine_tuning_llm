"""Évaluation d'un modèle sur les jeux de test — même protocole pour chaque modèle.

Mesures :
  - QCM (test MediQAl et FrenchMedMCQA) : exactitude (lettres exactement justes), score
    partiel pour les réponses multiples, taux de format (lettres extraites), sous-groupes
    (source, type, cas clinique, spécialités exclues de l'entraînement) ;
  - questions ouvertes (eval_clinique, MedQuAD) : ROUGE-L avec la référence, longueur, taux
    de réponses tronquées ; toutes les réponses sont sauvegardées pour relecture humaine ;
  - latence : cas cliniques traités un par un (référence avant optimisation par vLLM).

Protocole : prompt mis en forme comme à l'entraînement (configs/sft.yaml), génération
déterministe (greedy), arrêt sur <|im_end|>, modèle SFT chargé depuis le Hub à la révision
exacte de l'adaptateur puis fusionné. Tout est tracé dans MLflow (expérience chsa-eval).

Usage (sur le pod, environnement activé, depuis la racine du dépôt) :
    python -m src.evaluation.evaluate --modele sft --smoke
    python -m src.evaluation.evaluate --modele base
    python -m src.evaluation.evaluate --modele sft
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "eval.yaml"

# ---------------------------------------------------------------- mesures (fonctions pures)
_LETTER = r"[A-E](?![A-Za-zà-ÿœæŒÆ])"     # une lettre seule : pas le « E » de « Elle », ni le « C » de « Cœur »
_ANSWER = re.compile(rf"[Rr][ée]ponses?\s*:\s*({_LETTER}(?:\s*(?:,|;|et)\s*{_LETTER})*)")
_LEADING_LETTER = re.compile(r"^\s*([A-E])\s*[.)]")
_WORD = re.compile(r"\w+", re.UNICODE)


def parse_letters(text: str) -> list[str]:
    """Lettres de la réponse : « Réponse : D. … », « Réponses : B, C, D », sinon « D. … » en tête."""
    match = _ANSWER.search(text)
    if match:
        return sorted(set(re.findall(r"[A-E]", match.group(1))))
    match = _LEADING_LETTER.match(text)
    return [match.group(1)] if match else []


def partial_score(predicted: list[str], expected: list[str]) -> float:
    """Recouvrement |P ∩ G| / |P ∪ G| (1 si identiques, 0 si disjoints ou réponse vide)."""
    union = set(predicted) | set(expected)
    return len(set(predicted) & set(expected)) / len(union) if union else 0.0


def rouge_l(prediction: str, reference: str, max_words: int = 1000) -> float:
    """F1 de la plus longue sous-séquence commune de mots. Implémentation propre : la
    bibliothèque usuelle (rouge_score) supprime les caractères accentués du français."""
    pred = _WORD.findall(prediction.lower())[:max_words]
    ref = _WORD.findall(reference.lower())[:max_words]
    if not pred or not ref:
        return 0.0
    previous = [0] * (len(ref) + 1)
    for word in pred:
        current = [0]
        for j, ref_word in enumerate(ref, 1):
            current.append(previous[j - 1] + 1 if word == ref_word else max(previous[j], current[j - 1]))
        previous = current
    lcs = previous[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(pred), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else float("nan")


# ---------------------------------------------------------------- génération
def build_prompts(tokenizer, records: list[dict[str, Any]], template_kwargs: dict[str, Any]) -> list[str]:
    """Prompt identique à l'entraînement : messages system + user, chat template."""
    return [tokenizer.apply_chat_template(r["messages"][:-1], tokenize=False, add_generation_prompt=True,
                                          **template_kwargs) for r in records]


def generate(model, tokenizer, prompts: list[str], max_new_tokens: int, batch_size: int,
             eos_ids: list[int]) -> list[tuple[str, int]]:
    """Génération déterministe par lots (triés par longueur). Retourne (texte, nb de tokens générés)."""
    import torch
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]), reverse=True)
    results: list[tuple[str, int] | None] = [None] * len(prompts)
    for start in range(0, len(order), batch_size):
        batch = order[start:start + batch_size]
        encoded = tokenizer([prompts[i] for i in batch], return_tensors="pt", padding=True,
                            add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            output = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                    eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id)
        width = encoded["input_ids"].shape[1]
        for row, i in enumerate(batch):
            generated = output[row, width:]
            n_tokens = int((~torch.isin(generated, torch.tensor(eos_ids + [tokenizer.pad_token_id],
                                                                 device=generated.device))).sum())
            results[i] = (tokenizer.decode(generated, skip_special_tokens=True).strip(), n_tokens)
        print(f"    {min(start + batch_size, len(order))} / {len(order)}")
    return results  # type: ignore[return-value]


# ---------------------------------------------------------------- agrégation
def summarize_qcm(rows: list[dict[str, Any]], excluded_subjects: set[str]) -> dict[str, float]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups["qcm"].append(r)
        groups[f"qcm_{r['task']}"].append(r)
        groups[f"qcm_source_{r['source']}"].append(r)
        groups["qcm_avec_cas_clinique" if r["has_clinical_case"] else "qcm_sans_cas_clinique"].append(r)
        if r.get("medical_subject") in excluded_subjects:
            groups["qcm_specialites_hors_entrainement"].append(r)
    metrics = {}
    for name, items in groups.items():
        metrics[f"{name}_exactitude"] = mean([float(r["exact"]) for r in items])
        metrics[f"{name}_n"] = len(items)
    metrics["qcm_taux_format"] = mean([float(bool(r["predites"])) for r in rows])
    metrics["qcm_mcq_multi_score_partiel"] = mean([r["partiel"] for r in rows if r["task"] == "mcq_multi"])
    return metrics


def summarize_open(rows: list[dict[str, Any]], prefix: str, max_new_tokens: int) -> dict[str, float]:
    return {f"{prefix}_rouge_l": mean([r["rouge_l"] for r in rows]),
            f"{prefix}_longueur_tokens": mean([float(r["n_tokens"]) for r in rows]),
            f"{prefix}_taux_tronquees": mean([float(r["n_tokens"] >= max_new_tokens) for r in rows]),
            f"{prefix}_n": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modele", required=True, help="clé de la section « modeles » de la configuration")
    parser.add_argument("--smoke", action="store_true", help="quelques exemples par catégorie")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load((ROOT / cfg["config_entrainement"]).read_text(encoding="utf-8"))
    template_kwargs = train_cfg["model"].get("chat_template_kwargs") or {}
    model_cfg = cfg["modeles"][args.modele]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = f"eval-{args.modele}-{stamp}" + ("-smoke" if args.smoke else "")
    output_dir = Path(cfg["output_root"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    import mlflow
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(cfg["seed"])
    api = HfApi()

    # ---- Données (révision exacte)
    data_revision = api.dataset_info(cfg["dataset"]["repo_id"]).sha
    dataset = load_dataset(cfg["dataset"]["repo_id"], cfg["dataset"]["config"], revision=data_revision)
    test = list(dataset[cfg["dataset"]["test_split"]])
    qcm = [r for r in test if r["task"].startswith("mcq")]
    medquad = [r for r in test if r["task"] == "open_qa"]
    clinique = list(dataset[cfg["dataset"]["clinique_split"]])
    if args.smoke:
        qcm, medquad, clinique = (qcm[:cfg["smoke"]["qcm"]], medquad[:cfg["smoke"]["medquad"]],
                                  clinique[:cfg["smoke"]["clinique"]])
    print(f"QCM : {len(qcm)} — MedQuAD : {len(medquad)} — cas cliniques : {len(clinique)}")

    # ---- Modèle (révisions exactes), adaptateur fusionné si présent
    model_revision = api.model_info(model_cfg["id"]).sha
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["id"], revision=model_revision)
    tokenizer.padding_side = "left"                      # génération par lots
    model = AutoModelForCausalLM.from_pretrained(model_cfg["id"], revision=model_revision,
                                                 dtype=torch.bfloat16, device_map="cuda")
    adapter_revision = None
    if model_cfg.get("adaptateur"):
        from peft import PeftModel
        adapter_revision = api.model_info(model_cfg["adaptateur"]).sha
        model = PeftModel.from_pretrained(model, model_cfg["adaptateur"], revision=adapter_revision)
        model = model.merge_and_unload()
    model.eval()
    eos_ids = sorted({tokenizer.convert_tokens_to_ids(train_cfg["training"]["eos_token"]), tokenizer.eos_token_id})

    gen = cfg["generation"]
    metrics: dict[str, float] = {}
    predictions: list[dict[str, Any]] = []

    # ---- QCM
    print("[QCM] génération…")
    outputs = generate(model, tokenizer, build_prompts(tokenizer, qcm, template_kwargs),
                       gen["max_new_tokens_qcm"], gen["batch_size"], eos_ids)
    qcm_rows = []
    for record, (text, n_tokens) in zip(qcm, outputs):
        expected, predicted = parse_letters(record["messages"][-1]["content"]), parse_letters(text)
        qcm_rows.append({"id": record["id"], "source": record["source"], "task": record["task"],
                         "medical_subject": record.get("medical_subject"),
                         "has_clinical_case": record.get("has_clinical_case"),
                         "attendues": expected, "predites": predicted, "exact": predicted == expected,
                         "partiel": partial_score(predicted, expected), "n_tokens": n_tokens,
                         "prediction": text, "reference": record["messages"][-1]["content"]})
    metrics.update(summarize_qcm(qcm_rows, set(cfg["specialites_hors_entrainement"])))
    predictions += [{"categorie": "qcm", **r} for r in qcm_rows]

    # ---- Questions ouvertes
    for name, records in (("clinique", clinique), ("medquad", medquad)):
        if not records:
            continue
        print(f"[{name}] génération…")
        outputs = generate(model, tokenizer, build_prompts(tokenizer, records, template_kwargs),
                           gen["max_new_tokens_ouvert"], gen["batch_size"], eos_ids)
        rows = [{"id": r["id"], "source": r["source"], "medical_subject": r.get("medical_subject"),
                 "rouge_l": rouge_l(text, r["messages"][-1]["content"]), "n_tokens": n_tokens,
                 "prediction": text, "reference": r["messages"][-1]["content"]}
                for r, (text, n_tokens) in zip(records, outputs)]
        metrics.update(summarize_open(rows, name, gen["max_new_tokens_ouvert"]))
        predictions += [{"categorie": name, **r} for r in rows]

    # ---- Latence (un cas à la fois, comme un utilisateur)
    lat_cfg = cfg["latence"]
    sample = build_prompts(tokenizer, clinique[:lat_cfg["exemples"]], template_kwargs)
    durations, speeds = [], []
    generate(model, tokenizer, sample[:1], 16, 1, eos_ids)          # échauffement du GPU
    for prompt in sample:
        torch.cuda.synchronize()
        start = time.perf_counter()
        (_, n_tokens), = generate(model, tokenizer, [prompt], lat_cfg["max_new_tokens"], 1, eos_ids)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        durations.append(elapsed)
        speeds.append(n_tokens / elapsed if elapsed else 0.0)
    if durations:
        q = statistics.quantiles(durations, n=20) if len(durations) >= 2 else durations * 19
        metrics.update({"latence_mediane_s": round(statistics.median(durations), 3),
                        "latence_p95_s": round(q[18], 3),
                        "debit_tokens_par_s": round(statistics.median(speeds), 1)})

    # ---- Sauvegarde et MLflow
    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    (output_dir / "metriques.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    if mlflow.get_experiment_by_name(cfg["mlflow"]["experiment"]) is None:
        mlflow.create_experiment(cfg["mlflow"]["experiment"], artifact_location=cfg["mlflow"].get("artifact_location"))
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"modele": args.modele, "modele_id": model_cfg["id"], "modele_revision": model_revision,
                         "adaptateur": model_cfg.get("adaptateur") or "aucun",
                         "adaptateur_revision": adapter_revision or "aucune",
                         "dataset_revision": data_revision, "mode": "smoke" if args.smoke else "complet",
                         "gpu": torch.cuda.get_device_name(0)})
        mlflow.log_params({"batch_size": gen["batch_size"], "max_new_tokens_qcm": gen["max_new_tokens_qcm"],
                           "max_new_tokens_ouvert": gen["max_new_tokens_ouvert"], "decodage": "greedy",
                           **{f"template_{k}": v for k, v in template_kwargs.items()}})
        mlflow.log_metrics({k: v for k, v in metrics.items() if v == v})       # sans les NaN
        mlflow.log_artifact(str(predictions_path), artifact_path="predictions")
        mlflow.log_artifact(args.config, artifact_path="configuration")

    print("=" * 80 + f"\n{run_name}\n" + "=" * 80)
    for key in sorted(metrics):
        if not key.endswith("_n"):
            print(f"  {key:<45} {metrics[key]}")
    print(f"Prédictions : {predictions_path}")


if __name__ == "__main__":
    main()
