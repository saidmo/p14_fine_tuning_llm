"""Fine-tuning supervisé (SFT) de Qwen3-1.7B-Base avec LoRA — TRL + PEFT, suivi MLflow.

  - Données : configuration « sft » du dataset publié, à une révision exacte relevée au
    lancement. Le prompt (system + user) est mis en forme par le chat template EXACTEMENT
    comme à l'inférence (enable_thinking=False : le bloc <think> vide est dans le prompt) ;
    la completion est la réponse suivie de <|im_end|>. La loss ne porte que sur la réponse.
  - Modèle : révision exacte relevée au lancement ; LoRA sur toutes les couches linéaires,
    plus les vecteurs des tokens du chat template jamais entraînés dans le modèle Base
    (<|im_start|>, <|im_end|>, <think>, </think> : PEFT trainable_token_indices).
  - Contrôles : normes de ces vecteurs avant / après, génération de quelques réponses de
    validation (se terminent-elles par <|im_end|> ?) avant toute publication.
  - Traçabilité MLflow : commit Git (et modifications non commitées), révisions du dataset et
    du modèle, configuration, versions des bibliothèques, courbes, adaptateur final.
  - Sauvegarde : adaptateur sur le volume, puis poussé sur un dépôt PRIVÉ du Hub
    (rien d'important ne doit exister seulement sur le pod).

Usage (sur le pod, environnement activé, depuis la racine du dépôt) :
    python -m src.training.sft --smoke     # validation de la chaîne (quelques minutes)
    python -m src.training.sft             # entraînement complet
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "sft.yaml"


def git_info() -> dict[str, str]:
    def run(*cmd: str) -> str | None:
        try:
            return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    status = run("git", "status", "--porcelain")
    return {"git_commit": run("git", "rev-parse", "HEAD") or "inconnu",
            "git_modifications_non_commitees": "inconnu" if status is None else str(bool(status))}


def make_formatter(tokenizer, template_kwargs: dict[str, Any], eos_token: str):
    """[system, user, assistant] -> prompt (texte) / completion (texte).

    Le prompt est produit par le même appel au chat template qu'à l'inférence (vLLM) :
    il se termine par le bloc <think> vide quand enable_thinking=False. Le modèle n'apprend
    donc que la réponse, et l'entraînement voit le même texte que la production."""
    def to_prompt_completion(example: dict[str, Any]) -> dict[str, Any]:
        messages = example["messages"]
        prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False,
                                               add_generation_prompt=True, **template_kwargs)
        return {"prompt": prompt, "completion": messages[-1]["content"] + eos_token}
    return to_prompt_completion


def show_example(trainer, tokenizer) -> None:
    """Affiche un exemple tel que le modèle le voit, et la partie sur laquelle porte la loss.

    Selon la version de TRL, le masque est un champ completion_mask (1 = dans la loss) ou
    directement les labels (-100 = exclu de la loss)."""
    example = trainer.train_dataset[0]
    ids = example["input_ids"]
    if example.get("completion_mask") is not None:
        in_loss = [bool(m) for m in example["completion_mask"]]
    elif example.get("labels") is not None:
        in_loss = [label != -100 for label in example["labels"]]
    else:
        in_loss = None
    print("=" * 80 + "\nEXEMPLE D'ENTRAÎNEMENT (texte complet vu par le modèle)\n" + "=" * 80)
    print(tokenizer.decode(ids))
    if in_loss is None:
        print(f"(aucun masque trouvé : clés = {list(example.keys())})")
    else:
        first = in_loss.index(True) if True in in_loss else len(ids)
        target = [t for t, keep in zip(ids, in_loss) if keep]
        print("=" * 80 + f"\nPARTIE DONNÉE AU MODÈLE (hors loss) — {first} tokens, fin :\n" + "=" * 80)
        print(repr(tokenizer.decode(ids[max(0, first - 15):first])))
        print("=" * 80 + f"\nPARTIE SUR LAQUELLE PORTE LA LOSS — {len(target)} tokens sur {len(ids)} :\n" + "=" * 80)
        print(repr(tokenizer.decode(target)))
    print("=" * 80)


def token_norms(model, tokenizer, tokens: list[str]) -> dict[str, float]:
    """Norme du vecteur (partagé entrée / sortie) de chaque token : 0,375 = jamais entraîné."""
    import torch
    # Mesure EN SORTIE de la couche : avec PEFT, les vecteurs appris sont stockés à part et
    # appliqués au passage ; lire .weight afficherait les valeurs d'origine
    layer = model.get_input_embeddings()
    ids = tokenizer.convert_tokens_to_ids(tokens)
    device = next(model.parameters()).device
    with torch.no_grad():
        vectors = layer(torch.tensor(ids, device=device)).float()
    return {t: round(float(v.norm()), 3) for t, v in zip(tokens, vectors)}


def generation_check(model, tokenizer, records: list[dict[str, Any]], template_kwargs: dict[str, Any],
                     eos_token: str, max_new_tokens: int) -> float:
    """Génère quelques réponses de validation ; retourne la part terminée par le token de fin."""
    import torch
    eos_id = tokenizer.convert_tokens_to_ids(eos_token)
    model.eval()
    ended = 0
    for record in records:
        prompt = tokenizer.apply_chat_template(record["messages"][:-1], tokenize=False,
                                               add_generation_prompt=True, **template_kwargs)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                    eos_token_id=eos_id, pad_token_id=tokenizer.pad_token_id)
        generated = output[0, inputs["input_ids"].shape[1]:]
        finished = bool((generated == eos_id).any())
        ended += finished
        print("-" * 80 + f"\n[{record['task']}] terminée par {eos_token} : {finished} "
              f"({len(generated)} tokens)\n" + tokenizer.decode(generated)[:600])
    return ended / len(records) if records else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", help="run de validation de la chaîne")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    training_args = dict(cfg["training"])
    if args.smoke:
        training_args.update(cfg["smoke"]["training"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = f"sft-{stamp}" + ("-smoke" if args.smoke else "")
    output_dir = Path(cfg["output_root"]) / run_name

    # Imports lourds après l'analyse des arguments (aide immédiate avec --help)
    import mlflow
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(cfg["seed"])
    api = HfApi()

    # ---- Données (révision exacte)
    data_revision = api.dataset_info(cfg["dataset"]["repo_id"]).sha
    dataset = load_dataset(cfg["dataset"]["repo_id"], cfg["dataset"]["config"], revision=data_revision)
    train, evaluation = dataset[cfg["dataset"]["train_split"]], dataset[cfg["dataset"]["eval_split"]]
    # Exemples bruts pour le contrôle de génération : les premiers de chaque type de tâche
    per_type = cfg["controle_generation"]["exemples_par_type"]
    check_records = [r for task in ("mcq_single", "open_qa")
                     for r in [x for x in evaluation if x["task"] == task][:per_type]]
    if args.smoke:
        train = train.shuffle(seed=cfg["seed"]).select(range(cfg["smoke"]["max_train_samples"]))
        evaluation = evaluation.shuffle(seed=cfg["seed"]).select(range(cfg["smoke"]["max_eval_samples"]))

    # ---- Tokenizer (révision exacte), puis mise en forme des exemples
    model_revision = api.model_info(cfg["model"]["id"]).sha
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["id"], revision=model_revision)
    formatter = make_formatter(tokenizer, cfg["model"].get("chat_template_kwargs") or {},
                               training_args["eos_token"])
    train = train.map(formatter, remove_columns=train.column_names)
    evaluation = evaluation.map(formatter, remove_columns=evaluation.column_names)

    # ---- Modèle
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["id"], revision=model_revision, dtype=getattr(torch, cfg["model"]["dtype"]),
        attn_implementation=cfg["model"]["attn_implementation"])
    trainable_ids = tokenizer.convert_tokens_to_ids(cfg["trainable_tokens"])
    peft_config = LoraConfig(task_type="CAUSAL_LM", trainable_token_indices=trainable_ids, **cfg["lora"])
    norms_before = token_norms(model, tokenizer, cfg["trainable_tokens"])
    print(f"Normes avant entraînement : {norms_before}")

    # ---- MLflow (le callback de Transformers journalise dans le run ouvert ici)
    os.environ["MLFLOW_TRACKING_URI"] = cfg["mlflow"]["tracking_uri"]
    os.environ["MLFLOW_EXPERIMENT_NAME"] = cfg["mlflow"]["experiment"]
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    # L'emplacement des artefacts se fixe à la création de l'expérience : sans lui, MLflow écrit
    # dans ./mlruns, relatif au dossier de lancement (donc dans le dépôt Git)
    if mlflow.get_experiment_by_name(cfg["mlflow"]["experiment"]) is None:
        mlflow.create_experiment(cfg["mlflow"]["experiment"],
                                 artifact_location=cfg["mlflow"].get("artifact_location"))
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({**git_info(), "mode": "smoke" if args.smoke else "complet",
                         "dataset": cfg["dataset"]["repo_id"], "dataset_revision": data_revision,
                         "modele": cfg["model"]["id"], "modele_revision": model_revision,
                         "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__})
        mlflow.log_params({f"lora_{k}": (",".join(v) if isinstance(v, list) else v)
                           for k, v in cfg["lora"].items()})
        mlflow.log_params({"train_exemples": len(train), "eval_exemples": len(evaluation), "seed": cfg["seed"]})
        mlflow.log_artifact(args.config, artifact_path="configuration")
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
        mlflow.log_text(freeze, "environnement/pip_freeze.txt")

        sft_config = SFTConfig(output_dir=str(output_dir), run_name=run_name, report_to=["mlflow"],
                               seed=cfg["seed"], **training_args)
        trainer = SFTTrainer(model=model, args=sft_config, train_dataset=train, eval_dataset=evaluation,
                             processing_class=tokenizer, peft_config=peft_config)
        trainable, total = trainer.model.get_nb_trainable_parameters()
        print(f"Paramètres entraînables : {trainable:,} / {total:,} ({100 * trainable / total:.2f} %)")
        mlflow.log_params({"parametres_entrainables": trainable, "parametres_total": total})
        show_example(trainer, tokenizer)

        trainer.train()
        final = trainer.evaluate()

        norms_after = token_norms(trainer.model, tokenizer, cfg["trainable_tokens"])
        print(f"Normes avant : {norms_before}\nNormes après : {norms_after}")
        mlflow.log_dict({"avant": norms_before, "apres": norms_after}, "controles/normes_tokens.json")
        template_kwargs = cfg["model"].get("chat_template_kwargs") or {}
        ended = generation_check(trainer.model, tokenizer, check_records, template_kwargs,
                                 training_args["eos_token"], cfg["controle_generation"]["max_new_tokens"])
        mlflow.log_metric("controle_reponses_terminees", ended)
        print(f"Contrôle : {ended:.0%} des réponses se terminent par {training_args['eos_token']}")
        mlflow.log_metrics({f"final_{k}": v for k, v in final.items() if isinstance(v, (int, float))})
        print(f"Évaluation finale : {final}")

        adapter_dir = output_dir / "adapter"
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        mlflow.log_artifacts(str(adapter_dir), artifact_path="adapter")
        print(f"Adaptateur : {adapter_dir}")

        if cfg["hub"]["push"] and not args.smoke:
            message = f"{run_name} — dataset {data_revision[:10]} — eval_loss {final.get('eval_loss', float('nan')):.4f}"
            trainer.model.push_to_hub(cfg["hub"]["repo_id"], private=True, commit_message=message)
            tokenizer.push_to_hub(cfg["hub"]["repo_id"], private=True, commit_message=message)
            mlflow.set_tag("hub_repo", cfg["hub"]["repo_id"])
            print(f"Adaptateur publié (privé) : https://huggingface.co/{cfg['hub']['repo_id']}")


if __name__ == "__main__":
    main()
