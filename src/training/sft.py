"""Fine-tuning supervisé (SFT) de Qwen3-1.7B-Base avec LoRA — TRL + PEFT, suivi MLflow.

  - Données : configuration « sft » du dataset publié, à une révision exacte relevée au
    lancement ; chaque exemple est converti au format prompt (system + user) / completion
    (assistant) : la loss ne porte que sur la réponse.
  - Modèle : révision exacte relevée au lancement ; LoRA sur toutes les couches linéaires.
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


def to_prompt_completion(example: dict[str, Any]) -> dict[str, Any]:
    """[system, user, assistant] -> prompt [system, user] / completion [assistant]."""
    messages = example["messages"]
    return {"prompt": messages[:-1], "completion": messages[-1:]}


def show_example(trainer, tokenizer) -> None:
    """Affiche un exemple tel que le modèle le voit, et la partie sur laquelle porte la loss."""
    example = trainer.train_dataset[0]
    ids = example["input_ids"]
    mask = example.get("completion_mask")
    print("=" * 80 + "\nEXEMPLE D'ENTRAÎNEMENT (texte complet vu par le modèle)\n" + "=" * 80)
    print(tokenizer.decode(ids))
    if mask is not None:
        target = [t for t, m in zip(ids, mask) if m]
        print("=" * 80 + f"\nPARTIE SUR LAQUELLE PORTE LA LOSS ({len(target)} tokens sur {len(ids)})\n" + "=" * 80)
        print(tokenizer.decode(target))
    else:
        print(f"(pas de completion_mask dans l'exemple : clés = {list(example.keys())})")
    print("=" * 80)


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
    if args.smoke:
        train = train.shuffle(seed=cfg["seed"]).select(range(cfg["smoke"]["max_train_samples"]))
        evaluation = evaluation.shuffle(seed=cfg["seed"]).select(range(cfg["smoke"]["max_eval_samples"]))
    train = train.map(to_prompt_completion, remove_columns=train.column_names)
    evaluation = evaluation.map(to_prompt_completion, remove_columns=evaluation.column_names)

    # ---- Modèle et tokenizer (révision exacte)
    model_revision = api.model_info(cfg["model"]["id"]).sha
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["id"], revision=model_revision)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["id"], revision=model_revision, dtype=getattr(torch, cfg["model"]["dtype"]),
        attn_implementation=cfg["model"]["attn_implementation"])
    peft_config = LoraConfig(task_type="CAUSAL_LM", **cfg["lora"])

    # ---- MLflow (le callback de Transformers journalise dans le run ouvert ici)
    os.environ["MLFLOW_TRACKING_URI"] = cfg["mlflow"]["tracking_uri"]
    os.environ["MLFLOW_EXPERIMENT_NAME"] = cfg["mlflow"]["experiment"]
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
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
