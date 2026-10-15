"""Complément de relecture : reporter la relecture de sécurité sur le modèle DPO servi (non fusionné).

La relecture en aveugle a porté sur le DPO fusionné en bf16. Pour les mêmes 20 cas, on compare
sa réponse à celle du DPO non fusionné :
  - réponse identique : le verdict déjà donné reste valable ;
  - réponse différente : elle est à relire, avec la même grille et les mêmes règles.

Deux temps :
  1. sans option : écrit data/relecture/grille_complement.csv (réponses modifiées seulement ; ni
     l'ancienne réponse ni l'ancien verdict n'y figurent, pour ne pas influencer la relecture) ;
  2. --agreger, une fois la grille remplie : verdicts du DPO servi (repris ou nouveaux), comparés
     à ceux du SFT -> reports/relecture_securite_dpo_separe.json.

Usage :
    python -m src.evaluation.relecture_complement \\
        --dpo-fusionne /workspace/outputs/eval/eval-dpo-20261013-170100/predictions.jsonl \\
        --dpo-separe   /workspace/outputs/eval/eval-dpo_separe-20261015-145557/predictions.jsonl
    python -m src.evaluation.relecture_complement --dpo-fusionne … --dpo-separe … --agreger
"""
from __future__ import annotations

import argparse
import csv
import json

from src.evaluation.agreger_relecture import RELECTURE, ROOT, bilan, lire_verdicts, verdict
from src.evaluation.comparer_sft_dpo import load

COMPLEMENT = RELECTURE / "grille_complement.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dpo-fusionne", required=True, help="predictions.jsonl du DPO fusionné (relu)")
    parser.add_argument("--dpo-separe", required=True, help="predictions.jsonl du DPO non fusionné (servi)")
    parser.add_argument("--agreger", action="store_true", help="agréger la grille complémentaire remplie")
    args = parser.parse_args()

    verdicts = lire_verdicts(RELECTURE / "grille_relecture.csv", RELECTURE / "cle_relecture.json")
    fusionne, separe = load(args.dpo_fusionne)["clinique"], load(args.dpo_separe)["clinique"]
    modifies = [i for i in verdicts if fusionne[i]["prediction"] != separe[i]["prediction"]]
    print(f"Réponses DPO identiques : {len(verdicts) - len(modifies)} / {len(verdicts)} ; "
          f"à relire : {len(modifies)}")

    if not args.agreger:
        if COMPLEMENT.exists():
            raise SystemExit(f"{COMPLEMENT} existe déjà : remplir puis relancer avec --agreger")
        with open(RELECTURE / "grille_relecture.csv", encoding="utf-8-sig", newline="") as f:
            source = {r["id"]: r for r in csv.DictReader(f, delimiter=";")}
        with COMPLEMENT.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["id", "question", "reference", "reponse", "verdict", "commentaire"])
            for i in modifies:
                writer.writerow([i, source[i]["question"], source[i]["reference"],
                                 separe[i]["prediction"], "", ""])
        print(f"Grille complémentaire : {COMPLEMENT}")
        return

    with COMPLEMENT.open(encoding="utf-8-sig", newline="") as f:
        nouveaux = {r["id"]: verdict(r["verdict"], r["id"]) for r in csv.DictReader(f, delimiter=";")}
    if set(nouveaux) != set(modifies):
        raise SystemExit("La grille complémentaire ne correspond pas aux réponses modifiées")
    servis = {i: {"sft": v["sft"], "dpo": nouveaux.get(i, v["dpo"])} for i, v in verdicts.items()}
    resultats = {**bilan(servis), "modele_dpo": "non fusionné (adaptateur appliqué à l'exécution)",
                 "reponses_reprises": len(verdicts) - len(modifies), "reponses_relues": len(modifies),
                 "relecture_complementaire": "réponse DPO seule, sans l'ancienne réponse ni l'ancien verdict",
                 "verdicts_dpo_fusionne": bilan(verdicts)["verdicts"]["dpo"]}
    print(json.dumps(resultats, ensure_ascii=False, indent=2))
    report = ROOT / "reports" / "relecture_securite_dpo_separe.json"
    report.write_text(json.dumps(resultats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Rapport : {report}")


if __name__ == "__main__":
    main()
