"""Alignement par préférences (DPO) du modèle SFT fusionné, avec LoRA — TRL + PEFT, suivi MLflow.

  - Modèle : SFT fusionné (src/training/fusionner.py), à une révision ÉPINGLÉE. Il est le point
    de départ ET le modèle de référence : avec PEFT et sans ref_model, TRL calcule les
    log-probabilités de référence en désactivant l'adaptateur.
  - Données : configuration « dpo » du dataset, révision épinglée. Le prompt est mis en forme par
    le script EXACTEMENT comme au SFT et à l'inférence (enable_thinking=False, bloc <think> vide
    dans le prompt) ; chosen / rejected = réponse + <|im_end|>, déjà présent, donc TRL n'ajoute
    rien. Contrôle des longueurs avant l'entraînement (fermé par défaut).
  - Contrôles : génération sur les mêmes exemples avant / après (format QCM, fin de réponse,
    longueur) ; vecteurs des tokens du chat template inchangés (gelés).
  - Test DPO : mesuré une seule fois, après la sélection du checkpoint sur la validation, par un
    calcul indépendant de TRL (récompense implicite de chaque paire), par type de label et par
    origine. Le même calcul sur la validation est comparé à la métrique de TRL.
  - Traçabilité MLflow et publication de l'adaptateur sur un dépôt PRIVÉ du Hub.

Usage (sur le pod, environnement activé, depuis la racine du dépôt) :
    python -m src.training.dpo --smoke     # validation de la chaîne (quelques minutes)
    python -m src.training.dpo             # entraînement complet
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.evaluation.evaluate import generate, mean, parse_letters
from src.training.sft import git_info, token_norms

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "dpo.yaml"
TOKENS_TEMPLATE = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]


def make_formatter(tokenizer, template_kwargs: dict[str, Any], eos_token: str):
    """Paire conversationnelle -> prompt (texte) / chosen / rejected (texte + fin de réponse)."""
    def to_text(example: dict[str, Any]) -> dict[str, str]:
        prompt = tokenizer.apply_chat_template(example["prompt"], tokenize=False,
                                               add_generation_prompt=True, **template_kwargs)
        return {"prompt": prompt,
                "chosen": example["chosen"][-1]["content"] + eos_token,
                "rejected": example["rejected"][-1]["content"] + eos_token}
    return to_text


def n_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def check_lengths(tokenizer, rows: list[dict[str, str]], max_length: int) -> int:
    """Nombre de paires dont prompt + réponse la plus longue dépasse max_length."""
    return sum(n_tokens(tokenizer, r["prompt"]) + max(n_tokens(tokenizer, r["chosen"]),
                                                      n_tokens(tokenizer, r["rejected"])) > max_length
               for r in rows)


def show_example(trainer, tokenizer, eos_id: int) -> None:
    """Exemple tel que TRL l'a tokenisé : début du prompt, fin de chaque réponse.
    Vérifie qu'une réponse se termine par UN seul <|im_end|> (pas d'EOS ajouté en double)."""
    example = trainer.train_dataset[0]
    print("=" * 80 + f"\nEXEMPLE TOKENISÉ PAR TRL — clés : {list(example.keys())}\n" + "=" * 80)
    for key, ids in example.items():
        if not (key.endswith("ids") and isinstance(ids, list) and ids and isinstance(ids[0], int)):
            continue
        print(f"[{key}] {len(ids)} tokens — fin : {tokenizer.decode(ids[-12:])!r}")
        if key.startswith(("chosen", "rejected")) and ids[-1] == eos_id and len(ids) > 1 and ids[-2] == eos_id:
            raise SystemExit(f"{key} se termine par deux <|im_end|> : EOS ajouté en double")
    print("=" * 80)


def generation_check(model, tokenizer, cases: list[dict[str, Any]], template_kwargs: dict[str, Any],
                     eos_ids: list[int], gen: dict[str, Any], label: str) -> dict[str, float]:
    """Mêmes réglages que l'évaluation : QCM 64 tokens sans pénalité, ouvert avec pénalité."""
    model.eval()
    side, tokenizer.padding_side = tokenizer.padding_side, "left"
    metrics: dict[str, float] = {}
    for kind in ("qcm", "ouvert"):
        subset = [c for c in cases if (c["tache"].startswith("mcq")) == (kind == "qcm")]
        prompts = [tokenizer.apply_chat_template(c["prompt"], tokenize=False, add_generation_prompt=True,
                                                 **template_kwargs) for c in subset]
        max_new = gen["qcm"] if kind == "qcm" else gen["ouvert"]
        outputs = generate(model, tokenizer, prompts, max_new, gen["batch_size"], eos_ids,
                           repetition_penalty=1.0 if kind == "qcm" else gen["penalite"], verbose=False)
        for case, (text, n) in zip(subset, outputs):
            print("-" * 80 + f"\n[{label}] [{case['tache']}] {n} tokens"
                  + (f" — lettres : {parse_letters(text)}" if kind == "qcm" else "") + f"\n{text[:600]}")
        if kind == "qcm":
            metrics["format_qcm"] = mean([float(bool(parse_letters(t))) for t, _ in outputs])
        else:
            metrics["ouvert_terminees"] = mean([float(n < max_new) for _, n in outputs])
            metrics["ouvert_tokens_moyen"] = mean([float(n) for _, n in outputs])
    tokenizer.padding_side = side
    return metrics


def completion_logps(model, tokenizer, prompt: str, completion: str) -> float:
    """Somme des log-probabilités des tokens de la réponse. Prompt et réponse sont tokenisés
    séparément, sans tokens spéciaux ajoutés, comme dans TRL et comme à l'inférence."""
    import torch
    p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    c = tokenizer(completion, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([p + c], device=model.device)
    with torch.inference_mode():
        logits = model(input_ids=ids).logits[0, len(p) - 1:-1].float()
    return float(torch.log_softmax(logits, -1).gather(-1, ids[0, len(p):, None]).sum())


def preference_rewards(model, tokenizer, rows: list[dict[str, Any]], beta: float) -> list[dict[str, Any]]:
    """Récompense implicite DPO de chaque paire : beta x (écart à la référence de chosen
    - écart à la référence de rejected). Paire « préférée » si la marge est > 0.
    Référence = même modèle, adaptateur désactivé (le modèle SFT)."""
    model.eval()
    results = []
    for i, r in enumerate(rows, 1):
        pol = [completion_logps(model, tokenizer, r["prompt"], r[k]) for k in ("chosen", "rejected")]
        with model.disable_adapter():
            ref = [completion_logps(model, tokenizer, r["prompt"], r[k]) for k in ("chosen", "rejected")]
        chosen, rejected = beta * (pol[0] - ref[0]), beta * (pol[1] - ref[1])
        results.append({"id": r["id"], "origine": r["origine"], "label_type": r["label_type"],
                        "recompense_chosen": chosen, "recompense_rejected": rejected,
                        "marge": chosen - rejected, "preferee": chosen > rejected,
                        "logps_chosen": pol[0], "logps_rejected": pol[1],
                        "ref_logps_chosen": ref[0], "ref_logps_rejected": ref[1]})
        if i % 100 == 0:
            print(f"    {i} / {len(rows)}")
    return results


def summarize_rewards(results: list[dict[str, Any]], prefix: str, train_origins: set[str]) -> dict[str, float]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        groups["tout"].append(r)
        groups[r["label_type"]].append(r)
        groups["origines_entrainement" if r["origine"] in train_origins else "autres_origines"].append(r)
    metrics = {}
    for name, rows in groups.items():
        metrics[f"{prefix}_precision_{name}"] = mean([float(r["preferee"]) for r in rows])
        metrics[f"{prefix}_marge_{name}"] = mean([r["marge"] for r in rows])
        metrics[f"{prefix}_precision_{name}_n"] = len(rows)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", help="run de validation de la chaîne")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    sft_cfg = yaml.safe_load((ROOT / cfg["config_sft"]).read_text(encoding="utf-8"))
    eval_cfg = yaml.safe_load((ROOT / cfg["config_eval"]).read_text(encoding="utf-8"))
    template_kwargs = sft_cfg["model"].get("chat_template_kwargs") or {}
    eos_token = sft_cfg["training"]["eos_token"]
    training_args = dict(cfg["training"])
    if args.smoke:
        training_args.update(cfg["smoke"]["training"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = f"dpo-{stamp}" + ("-smoke" if args.smoke else "")
    output_dir = Path(cfg["output_root"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    import mlflow
    import torch
    import trl
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import DPOConfig, DPOTrainer

    set_seed(cfg["seed"])
    api = HfApi()
    model_id, model_rev = cfg["model"]["id"], cfg["model"]["revision"]
    data_cfg = cfg["dataset"]

    # ---- Tokenizer du modèle SFT fusionné : il porte la fin de réponse apprise
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_rev)
    if tokenizer.eos_token != eos_token:
        raise SystemExit(f"eos_token = {tokenizer.eos_token!r}, attendu {eos_token!r} : TRL ajouterait "
                         "un autre token de fin aux réponses")
    eos_ids = sorted({tokenizer.convert_tokens_to_ids(eos_token), tokenizer.eos_token_id,
                      tokenizer.convert_tokens_to_ids("<|endoftext|>")})

    # ---- Données (révision épinglée)
    dataset = load_dataset(data_cfg["repo_id"], data_cfg["config"], revision=data_cfg["revision"])
    train, evaluation, test = (dataset[data_cfg[k]] for k in ("train_split", "eval_split", "test_split"))
    if args.smoke:
        train = train.shuffle(seed=cfg["seed"]).select(range(cfg["smoke"]["max_train_samples"]))
        evaluation = evaluation.shuffle(seed=cfg["seed"]).select(range(cfg["smoke"]["max_eval_samples"]))
        test = test.shuffle(seed=cfg["seed"]).select(range(cfg["smoke"]["max_test_samples"]))
    train_origins = set(train["origine"])
    formatter = make_formatter(tokenizer, template_kwargs, eos_token)

    # Exemples raw du test et de la validation pour le calcul indépendant (métadonnées conservées)
    test_rows = [{**formatter(r), "id": r["id"], "origine": r["origine"], "label_type": r["label_type"]}
                 for r in test]
    val_rows = [{**formatter(r), "id": r["id"], "origine": r["origine"], "label_type": r["label_type"]}
                for r in evaluation]
    train = train.map(formatter, remove_columns=train.column_names)
    evaluation = evaluation.map(formatter, remove_columns=evaluation.column_names)

    # ---- Longueurs (fermé par défaut pour l'entraînement ; compté pour le test)
    max_length = training_args["max_length"]
    over = {"train": check_lengths(tokenizer, list(train), max_length),
            "validation": check_lengths(tokenizer, val_rows, max_length),
            "test": check_lengths(tokenizer, test_rows, max_length)}
    print(f"Paires au-delà de {max_length} tokens : {over}")
    if over["train"] or over["validation"]:
        raise SystemExit("Paires d'entraînement ou de validation tronquées : vérifier max_length")

    # ---- Contrôle de génération : QCM et questions ouvertes (validation SFT), prompts DPO
    gen_cfg = cfg["controle_generation"]
    sft_val = load_dataset(data_cfg["repo_id"], "sft", split="validation", revision=data_cfg["revision"])
    per_type = gen_cfg["exemples_par_type"]
    cases = [{"tache": task, "prompt": r["messages"][:-1]}
             for task in ("mcq_single", "mcq_multi", "open_qa")
             for r in [x for x in sft_val if x["task"] == task][:per_type]]
    dpo_val = dataset[data_cfg["eval_split"]]
    cases += [{"tache": f"dpo_{o}", "prompt": next(r for r in dpo_val if r["origine"] == o)["prompt"]}
              for o in gen_cfg["origines_dpo"]]
    gen = {"qcm": eval_cfg["generation"]["max_new_tokens_qcm"], "ouvert": gen_cfg["max_new_tokens_ouvert"],
           "penalite": eval_cfg["generation"]["repetition_penalty_ouvert"], "batch_size": gen_cfg["batch_size"]}

    # ---- Modèle SFT fusionné ; adaptateur LoRA sans tokens entraînables
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=model_rev,
                                                 dtype=getattr(torch, cfg["model"]["dtype"]),
                                                 attn_implementation=cfg["model"]["attn_implementation"]).to("cuda")
    # revision : inscrite dans adapter_config.json, avec base_model_name_or_path = modèle SFT fusionné
    peft_config = LoraConfig(task_type="CAUSAL_LM", revision=model_rev, **cfg["lora"])
    norms_before = token_norms(model, tokenizer, TOKENS_TEMPLATE)
    print(f"Normes des tokens du chat template (modèle de départ) : {norms_before}")
    print("[contrôle AVANT] génération…")
    check_before = generation_check(model, tokenizer, cases, template_kwargs, eos_ids, gen, "AVANT")

    # ---- MLflow
    os.environ["MLFLOW_TRACKING_URI"] = cfg["mlflow"]["tracking_uri"]
    os.environ["MLFLOW_EXPERIMENT_NAME"] = cfg["mlflow"]["experiment"]
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    if mlflow.get_experiment_by_name(cfg["mlflow"]["experiment"]) is None:
        mlflow.create_experiment(cfg["mlflow"]["experiment"],
                                 artifact_location=cfg["mlflow"].get("artifact_location"))
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({**git_info(), "mode": "smoke" if args.smoke else "complet",
                         "dataset": data_cfg["repo_id"], "dataset_revision": data_cfg["revision"],
                         "modele": model_id, "modele_revision": model_rev,
                         "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                         "trl": trl.__version__})
        mlflow.log_params({f"lora_{k}": (",".join(v) if isinstance(v, list) else v)
                           for k, v in cfg["lora"].items()})
        mlflow.log_params({"train_paires": len(train), "eval_paires": len(evaluation),
                           "test_paires": len(test_rows), "seed": cfg["seed"],
                           "test_paires_tronquees": over["test"]})
        mlflow.log_metrics({f"controle_avant_{k}": v for k, v in check_before.items()})
        mlflow.log_artifact(args.config, artifact_path="configuration")
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
        mlflow.log_text(freeze, "environnement/pip_freeze.txt")

        dpo_config = DPOConfig(output_dir=str(output_dir), run_name=run_name, report_to=["mlflow"],
                               seed=cfg["seed"], **training_args)
        trainer = DPOTrainer(model=model, ref_model=None, args=dpo_config, train_dataset=train,
                             eval_dataset=evaluation, processing_class=tokenizer, peft_config=peft_config)
        trainable, total = trainer.model.get_nb_trainable_parameters()
        print(f"Paramètres entraînables : {trainable:,} / {total:,} ({100 * trainable / total:.2f} %)")
        mlflow.log_params({"parametres_entrainables": trainable, "parametres_total": total})
        show_example(trainer, tokenizer, tokenizer.convert_tokens_to_ids(eos_token))

        trainer.train()
        final = trainer.evaluate()          # meilleur checkpoint (validation)
        mlflow.log_metrics({f"final_{k}": v for k, v in final.items() if isinstance(v, (int, float))})
        print(f"Évaluation finale (validation, TRL) : {final}")

        # ---- Contrôles après entraînement
        norms_after = token_norms(trainer.model, tokenizer, TOKENS_TEMPLATE)
        print(f"Normes avant : {norms_before}\nNormes après : {norms_after}")
        mlflow.log_dict({"avant": norms_before, "apres": norms_after}, "controles/normes_tokens.json")
        if norms_after != norms_before:
            raise SystemExit("Vecteurs des tokens du chat template modifiés alors qu'ils sont gelés")
        print("[contrôle APRÈS] génération…")
        check_after = generation_check(trainer.model, tokenizer, cases, template_kwargs, eos_ids, gen, "APRÈS")
        mlflow.log_metrics({f"controle_apres_{k}": v for k, v in check_after.items()})
        print(f"Contrôle de génération — avant : {check_before}\n                         après : {check_after}")

        # ---- Récompenses implicites : validation (comparée à TRL), puis test, UNE fois
        beta = training_args["beta"]
        print("[validation] récompenses implicites…")
        val_results = preference_rewards(trainer.model, tokenizer, val_rows, beta)
        print("[test] récompenses implicites…")
        test_results = preference_rewards(trainer.model, tokenizer, test_rows, beta)
        metrics = {**summarize_rewards(val_results, "validation", train_origins),
                   **summarize_rewards(test_results, "test", train_origins)}
        # Comparaison avec TRL sur la validation. Les MOYENNES (log-probabilités, marge) vérifient
        # la définition du calcul ; la précision, elle, est sensible au bruit numérique bf16
        # (lots avec padding chez TRL, séquences seules ici) quand les marges sont proches de 0
        comparaison = {"logps_chosen": (mean([r["logps_chosen"] for r in val_results]),
                                        final.get("eval_logps/chosen")),
                       "logps_rejected": (mean([r["logps_rejected"] for r in val_results]),
                                          final.get("eval_logps/rejected")),
                       "marge": (metrics["validation_marge_tout"], final.get("eval_rewards/margins")),
                       "precision": (metrics["validation_precision_tout"], final.get("eval_rewards/accuracies"))}
        metrics.update({f"validation_{k}_moyenne": v[0] for k, v in comparaison.items()
                        if k.startswith("logps")})
        mlflow.log_metrics(metrics)
        mlflow.log_dict({k: {"independant": v[0], "trl": v[1]} for k, v in comparaison.items()},
                        "controles/comparaison_trl_validation.json")
        print("Validation — calcul indépendant / TRL :")
        for k, (ours, theirs) in comparaison.items():
            print(f"  {k:<16} {ours:>10.4f}   {theirs}")
        for key in sorted(k for k in metrics if k.startswith("test_") and not k.endswith("_n")):
            print(f"  {key:<45} {metrics[key]}   (n={metrics.get(key.replace('_marge_', '_precision_') + '_n')})")
        results_path = output_dir / "recompenses.jsonl"
        with results_path.open("w", encoding="utf-8") as f:
            for split, rows in (("validation", val_results), ("test", test_results)):
                for r in rows:
                    f.write(json.dumps({"split": split, **r}, ensure_ascii=False) + "\n")
        mlflow.log_artifact(str(results_path), artifact_path="recompenses")

        # ---- Sauvegarde et publication (le résultat est publié et documenté, bon ou mauvais :
        # l'acceptation se décide à l'évaluation complète, selon les seuils fixés a priori)
        adapter_dir = output_dir / "adapter"
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        mlflow.log_artifacts(str(adapter_dir), artifact_path="adapter")
        print(f"Adaptateur : {adapter_dir}")

        if cfg["hub"]["push"] and not args.smoke:
            repo = cfg["hub"]["repo_id"]
            api.create_repo(repo, private=True, exist_ok=True)
            if not api.model_info(repo).private:
                raise SystemExit(f"{repo} est PUBLIC : publication refusée")
            message = (f"{run_name} — sft {model_rev[:10]} — dataset {data_cfg['revision'][:10]} — "
                       f"eval_loss {final.get('eval_loss', float('nan')):.4f}")
            commit = api.upload_folder(repo_id=repo, folder_path=str(adapter_dir), commit_message=message)
            mlflow.set_tags({"hub_repo": repo, "hub_revision": commit.oid})
            print(f"Adaptateur publié (privé) : https://huggingface.co/{repo}\nRÉVISION : {commit.oid}")


if __name__ == "__main__":
    main()
