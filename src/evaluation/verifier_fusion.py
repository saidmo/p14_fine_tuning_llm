"""Fidélité de la fusion d'un adaptateur LoRA dans des poids bf16 : quelle part de la mise à jour
survit à l'arrondi, et le comportement appris est-il conservé ?

Contexte : en bf16, un poids n'a que 8 bits de mantisse. Une mise à jour LoRA très petite devant
le poids (DPO à lr 5e-6) peut être en partie effacée par l'arrondi de W + BA, alors qu'elle est
bien appliquée quand l'adaptateur reste séparé (calcul W·x + B·A·x, comme à l'entraînement).

1. Poids (sans données) : pour chaque couche LoRA, Δ = scaling · B·A (fp32) comparé à la
   variation réellement obtenue après fusion bf16 : erreur relative ||(W_fusionné - W) - Δ|| / ||Δ||
   et part des coefficients où la mise à jour a été entièrement effacée.
2. Préférences (option --preferences, pour un adaptateur DPO) : précision et marge des
   récompenses implicites sur la validation DPO, adaptateur séparé contre modèle fusionné, même
   référence (adaptateur désactivé), et accord du signe paire par paire.

Usage :
    python -m src.evaluation.verifier_fusion --base saidmo2/chsa-qwen3-1.7b-sft --base-revision <sha> \\
        --adaptateur saidmo2/chsa-qwen3-1.7b-sft-dpo-lora --adaptateur-revision <sha> --preferences
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from src.evaluation.evaluate import mean
from src.training.dpo import completion_logps, make_formatter

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for arg in ("--base", "--base-revision", "--adaptateur", "--adaptateur-revision"):
        parser.add_argument(arg, required=True)
    parser.add_argument("--preferences", action="store_true", help="test des préférences (adaptateur DPO)")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from peft.tuners.lora import LoraLayer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.adaptateur, revision=args.adaptateur_revision)
    model = AutoModelForCausalLM.from_pretrained(args.base, revision=args.base_revision,
                                                 dtype=torch.bfloat16).to("cuda")
    peft_model = PeftModel.from_pretrained(model, args.adaptateur, revision=args.adaptateur_revision).eval()
    merged = copy.deepcopy(peft_model).merge_and_unload().eval()
    merged_weights = dict(merged.named_parameters())

    # ---- 1. Poids
    err2 = delta2 = w2 = 0.0
    nonzero = erased = 0
    for name, module in peft_model.named_modules():
        if not isinstance(module, LoraLayer) or "default" not in module.lora_A:
            continue
        w = module.base_layer.weight.float()
        delta = (module.lora_B["default"].weight.float() @ module.lora_A["default"].weight.float()) \
            * module.scaling["default"]
        key = name.removeprefix("base_model.model.") + ".weight"
        variation = merged_weights[key].float() - w
        err2 += float(((variation - delta) ** 2).sum())
        delta2 += float((delta ** 2).sum())
        w2 += float((w ** 2).sum())
        mask = delta != 0
        nonzero += int(mask.sum())
        erased += int((mask & (variation == 0)).sum())
    poids = {"norme_relative_mise_a_jour": (delta2 / w2) ** 0.5,
             "erreur_relative_fusion": (err2 / delta2) ** 0.5,
             "part_coefficients_effaces": erased / nonzero}
    print(f"Poids : {json.dumps(poids, indent=2)}")
    resultats: dict = {"base": args.base, "base_revision": args.base_revision,
                       "adaptateur": args.adaptateur, "adaptateur_revision": args.adaptateur_revision,
                       "poids": poids}

    # ---- 2. Préférences sur la validation DPO (adaptateur séparé / fusionné, même référence)
    if args.preferences:
        from datasets import load_dataset
        dpo_cfg = yaml.safe_load((ROOT / "configs" / "dpo.yaml").read_text(encoding="utf-8"))
        sft_cfg = yaml.safe_load((ROOT / dpo_cfg["config_sft"]).read_text(encoding="utf-8"))
        d = dpo_cfg["dataset"]
        val = load_dataset(d["repo_id"], d["config"], split=d["eval_split"], revision=d["revision"])
        fmt = make_formatter(tokenizer, sft_cfg["model"].get("chat_template_kwargs") or {},
                             sft_cfg["training"]["eos_token"])
        beta = dpo_cfg["training"]["beta"]
        marges = {"separe": [], "fusionne": []}
        for i, row in enumerate(map(fmt, val), 1):
            lp = {k: completion_logps(peft_model, tokenizer, row["prompt"], row[k]) for k in ("chosen", "rejected")}
            lm = {k: completion_logps(merged, tokenizer, row["prompt"], row[k]) for k in ("chosen", "rejected")}
            with peft_model.disable_adapter():
                ref = {k: completion_logps(peft_model, tokenizer, row["prompt"], row[k]) for k in ("chosen", "rejected")}
            for nom, pol in (("separe", lp), ("fusionne", lm)):
                marges[nom].append(beta * ((pol["chosen"] - ref["chosen"]) - (pol["rejected"] - ref["rejected"])))
            if i % 100 == 0:
                print(f"    {i} / {len(val)}")
        resultats["preferences_validation"] = {
            "n": len(marges["separe"]),
            **{f"precision_{k}": mean([float(m > 0) for m in v]) for k, v in marges.items()},
            **{f"marge_{k}": mean(v) for k, v in marges.items()},
            "accord_du_signe": mean([float((a > 0) == (b > 0)) for a, b in zip(marges["separe"], marges["fusionne"])]),
        }
        print(f"Préférences (validation) : {json.dumps(resultats['preferences_validation'], indent=2)}")

    nom = args.adaptateur.split("/")[-1]
    report = ROOT / "reports" / f"verification_fusion_{nom}.json"
    report.write_text(json.dumps(resultats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Rapport : {report}")


if __name__ == "__main__":
    main()
