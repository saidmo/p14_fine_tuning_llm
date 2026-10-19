"""Fusion d'un adaptateur LoRA dans son modèle de base, vérifications, publication (dépôt PRIVÉ).

Deux usages dans le projet, avec le même script :
  - semaine 3 : base Qwen3-1.7B-Base + adaptateur SFT  -> modèle SFT fusionné, point de départ
    ET modèle de référence du DPO ;
  - semaine 4 : modèle SFT fusionné + adaptateur DPO -> fusion REFUSÉE par le contrôle 4
     (accord top-1 95,3 % < 98 %) : l'adaptateur DPO est servi séparé par vLLM
     (diagnostic : src/evaluation/verifier_fusion.py, reports/verification_fusion_*.json).

Fermé par défaut : les deux révisions doivent être données explicitement (SHA de 40 caractères),
et rien n'est publié si une vérification échoue :
  1. le tokenizer de l'adaptateur s'arrête sur <|im_end|> (sinon le modèle ne s'arrêterait pas
     en production) ;
  2. les vecteurs des tokens du chat template sont identiques avant et après la fusion ;
  3. les poids d'entrée et de sortie restent liés (tie_word_embeddings) ;
  4. le modèle fusionné prédit comme le modèle avec adaptateur (logits sur un prompt réel) ;
  5. le modèle relu depuis le disque est identique, tenseur par tenseur, au modèle fusionné.
La provenance (révisions, commit Git, versions, résultats des contrôles) est publiée avec les
poids dans provenance.json.

Usage (sur le pod, environnement activé, depuis la racine du dépôt) :
    python -m src.training.fusionner \\
        --base Qwen/Qwen3-1.7B-Base --base-revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1 \\
        --adaptateur saidmo2/chsa-qwen3-1.7b-sft-lora \\
        --adaptateur-revision a7c6bf2d639ae2ba1c39ae18c266c4107e897462 \\
        --repo saidmo2/chsa-qwen3-1.7b-sft
    (ajouter --sans-publication pour tout vérifier sans rien publier)
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.training.sft import git_info, token_norms

ROOT = Path(__file__).resolve().parents[2]
TOKENS_TEMPLATE = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
FIN_DE_REPONSE = "<|im_end|>"
SHA = re.compile(r"^[0-9a-f]{40}$")
# Seuil du contrôle 4 : en bf16, la fusion arrondit W + BA ; quelques égalités quasi parfaites
# entre deux tokens peuvent basculer, pas davantage
ACCORD_TOP1_MIN = 0.98


def vecteurs(model, ids: list[int]):
    """Vecteurs EN SORTIE de la couche d'embedding (avec PEFT, les vecteurs appris sont appliqués
    au passage et ne sont pas dans .weight)."""
    import torch
    layer = model.get_input_embeddings()
    with torch.no_grad():
        return layer(torch.tensor(ids, device=next(model.parameters()).device)).float().cpu()


def logits(model, inputs):
    import torch
    with torch.inference_mode():
        return model(**inputs).logits[0].float().cpu()


def poids_lies(model) -> bool:
    return model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--adaptateur", required=True)
    parser.add_argument("--adaptateur-revision", required=True)
    parser.add_argument("--repo", required=True, help="dépôt de destination (créé PRIVÉ)")
    parser.add_argument("--sortie", default="/workspace/outputs/fusion")
    parser.add_argument("--sans-publication", action="store_true")
    args = parser.parse_args()
    for nom, rev in (("--base-revision", args.base_revision), ("--adaptateur-revision", args.adaptateur_revision)):
        if not SHA.match(rev):
            raise SystemExit(f"{nom} doit être un SHA complet (40 caractères hexadécimaux) : {rev!r}")

    import peft
    import torch
    import transformers
    from huggingface_hub import HfApi
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.sortie) / f"{args.repo.split('/')[-1]}-{stamp}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    controles: dict[str, object] = {}

    # ---- 1. Tokenizer de l'adaptateur : c'est lui qui porte la fin de réponse apprise
    tokenizer = AutoTokenizer.from_pretrained(args.adaptateur, revision=args.adaptateur_revision)
    if tokenizer.eos_token != FIN_DE_REPONSE:
        raise SystemExit(f"eos_token du tokenizer = {tokenizer.eos_token!r}, attendu {FIN_DE_REPONSE!r}")
    ids = tokenizer.convert_tokens_to_ids(TOKENS_TEMPLATE)

    # ---- Modèle avec adaptateur (non fusionné)
    model = AutoModelForCausalLM.from_pretrained(args.base, revision=args.base_revision,
                                                 dtype=torch.bfloat16).to(device)
    model = PeftModel.from_pretrained(model, args.adaptateur, revision=args.adaptateur_revision).eval()
    peft_cfg = model.peft_config["default"]
    print(f"Adaptateur : r={peft_cfg.r}, alpha={peft_cfg.lora_alpha}, "
          f"base déclarée={peft_cfg.base_model_name_or_path}, "
          f"trainable_token_indices={peft_cfg.trainable_token_indices}")

    # Prompt réel, mis en forme comme à l'entraînement et à l'inférence
    fmt = yaml.safe_load((ROOT / "configs" / "format.yaml").read_text(encoding="utf-8"))
    sft_cfg = yaml.safe_load((ROOT / "configs" / "sft.yaml").read_text(encoding="utf-8"))
    messages = [{"role": "system", "content": fmt["system_prompt"].strip()},
                {"role": "user", "content": "Quels sont les signes cliniques d'une embolie pulmonaire ?"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           **(sft_cfg["model"].get("chat_template_kwargs") or {}))
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    vec_avant, logits_avant = vecteurs(model, ids), logits(model, inputs)
    normes = token_norms(model, tokenizer, TOKENS_TEMPLATE)

    # ---- Fusion
    merged = model.merge_and_unload().eval()
    vec_apres, logits_apres = vecteurs(merged, ids), logits(merged, inputs)

    # ---- 2. Vecteurs des tokens du chat template conservés
    ecart = float((vec_avant - vec_apres).abs().max())
    controles["ecart_max_vecteurs_tokens"] = ecart
    if ecart > 1e-6:
        raise SystemExit(f"Vecteurs des tokens du chat template modifiés par la fusion (écart {ecart})")
    # ---- 3. Entrée et sortie liées
    controles["poids_lies"] = poids_lies(merged)
    if not controles["poids_lies"]:
        raise SystemExit("Poids d'entrée et de sortie déliés après la fusion")
    # ---- 4. Même comportement que le modèle avec adaptateur
    accord = float((logits_avant.argmax(-1) == logits_apres.argmax(-1)).float().mean())
    controles["logits_ecart_max"] = round(float((logits_avant - logits_apres).abs().max()), 4)
    controles["logits_accord_top1"] = round(accord, 4)
    controles["logits_positions"] = int(logits_avant.shape[0])
    print(f"Normes des tokens : {normes}\nContrôles après fusion : {controles}")
    if accord < ACCORD_TOP1_MIN:
        raise SystemExit(f"Accord top-1 adaptateur / fusionné {accord:.3f} < {ACCORD_TOP1_MIN}")

    # ---- Arrêt de génération : <|im_end|> (réponse) et <|endoftext|> (fin de texte du modèle Base).
    # Lu par generate() et par vLLM (semaine 4)
    merged.generation_config.eos_token_id = sorted({tokenizer.convert_tokens_to_ids(FIN_DE_REPONSE),
                                                    tokenizer.convert_tokens_to_ids("<|endoftext|>")})
    merged.generation_config.pad_token_id = tokenizer.pad_token_id

    # ---- Sauvegarde, puis 5. relecture depuis le disque
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    relu = AutoModelForCausalLM.from_pretrained(str(out_dir), dtype=torch.bfloat16).to(device)
    attendu, obtenu = merged.state_dict(), relu.state_dict()
    differents = sorted(set(attendu) ^ set(obtenu)) + [k for k in attendu.keys() & obtenu.keys()
                                                         if not torch.equal(attendu[k], obtenu[k])]
    controles["relecture_tenseurs"] = len(attendu)
    controles["relecture_identique"] = not differents and poids_lies(relu)
    if not controles["relecture_identique"]:
        raise SystemExit(f"Modèle relu différent du modèle fusionné : {differents[:10]} "
                         f"(poids liés : {poids_lies(relu)})")
    print(f"Relecture : {len(attendu)} tenseurs identiques, poids liés")

    provenance = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base": args.base, "base_revision": args.base_revision,
        "adaptateur": args.adaptateur, "adaptateur_revision": args.adaptateur_revision,
        "adaptateur_config": {"r": peft_cfg.r, "lora_alpha": peft_cfg.lora_alpha,
                              "base_declaree": peft_cfg.base_model_name_or_path,
                              "trainable_token_indices": peft_cfg.trainable_token_indices},
        "dtype": "bfloat16", "normes_tokens_template": normes, "controles": controles,
        **git_info(),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "peft": peft.__version__},
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2,
                                                       default=str), encoding="utf-8")
    print(f"Modèle fusionné : {out_dir}")

    if args.sans_publication:
        print("--sans-publication : rien n'est publié.")
        return
    api = HfApi()
    api.create_repo(args.repo, private=True, exist_ok=True)
    if not api.model_info(args.repo).private:          # dépôt préexistant et public : on refuse
        raise SystemExit(f"{args.repo} est PUBLIC : publication refusée")
    message = (f"fusion {args.base}@{args.base_revision[:10]} + "
               f"{args.adaptateur}@{args.adaptateur_revision[:10]}")
    commit = api.upload_folder(repo_id=args.repo, folder_path=str(out_dir), commit_message=message)
    print(f"Publié (privé) : https://huggingface.co/{args.repo}\n"
          f"RÉVISION À ÉPINGLER : {commit.oid}")


if __name__ == "__main__":
    main()
