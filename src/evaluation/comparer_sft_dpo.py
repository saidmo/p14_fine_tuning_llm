"""Comparaison appariée des prédictions SFT et DPO, et grille de relecture de sécurité en aveugle.

Deux évaluations du même jeu de test (src.evaluation.evaluate, même protocole) sont comparées
exemple par exemple, sur les réponses libres (cas cliniques, MedQuAD) :
  - réponses tronquées (limite de 512 tokens) : tableau apparié et test exact de McNemar ;
  - longueur des réponses ;
  - cause des troncatures : boucle de répétition ou réponse simplement longue (heuristique) ;
  - langue de la réponse aux cas cliniques français (heuristique par mots-outils).
Puis 20 cas cliniques, tirés au hasard avec une graine fixée a priori, sont exportés pour une
relecture humaine EN AVEUGLE : les deux réponses sont nommées A et B dans un ordre aléatoire,
la correspondance est écrite dans un fichier séparé, à n'ouvrir qu'après la relecture.

Sorties :
  reports/comparaison_sft_dpo.json      agrégats uniquement (versionné)
  data/relecture/grille_relecture.csv   questions, références, réponses A / B, verdicts à saisir
  data/relecture/cle_relecture.json     correspondance A / B -> SFT / DPO (ne pas ouvrir avant)

Options : --rapport (nom du rapport), --sans-grille (comparaison seule). Une grille existante n'est
jamais écrasée.

Usage (sur le pod, depuis la racine du dépôt) :
    python -m src.evaluation.comparer_sft_dpo \\
        --sft /workspace/outputs/eval/eval-sft-<date>/predictions.jsonl \\
        --dpo /workspace/outputs/eval/eval-dpo-20261013-170100/predictions.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
from math import comb
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CATEGORIES = ("clinique", "medquad")
MAX_NEW_TOKENS = 512                     # = generation.max_new_tokens_ouvert (configs/eval.yaml)
GRAINE, N_RELECTURE = 42, 20             # fixés a priori
MOTS_FR = {"le", "la", "les", "de", "des", "du", "est", "et", "une", "un", "il", "pour", "dans",
           "que", "qui", "pas", "sur", "avec", "au", "aux", "chez", "sont"}
MOTS_EN = {"the", "of", "and", "is", "are", "to", "in", "with", "for", "this", "that", "be", "it",
           "on", "as", "or", "should", "may"}


def load(path: str) -> dict[str, dict[str, dict[str, Any]]]:
    """{catégorie: {id: prédiction}} pour les réponses libres."""
    out: dict[str, dict[str, dict[str, Any]]] = {c: {} for c in CATEGORIES}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["categorie"] in out:
            out[row["categorie"]][row["id"]] = row
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """p-valeur bilatérale exacte : sous H0, les b + c désaccords se répartissent à 50 / 50."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def words(text: str) -> list[str]:
    return re.findall(r"[a-zàâçéèêëîïôûùüÿœ']+", text.lower())


def langue(text: str) -> str:
    w = words(text)
    fr, en = sum(x in MOTS_FR for x in w), sum(x in MOTS_EN for x in w)
    return "en" if en > fr else "fr"


def boucle(text: str, fenetre: int = 200) -> bool:
    """Heuristique : dans les 200 derniers mots, moins de 70 % de 4-grammes distincts."""
    w = words(text)[-fenetre:]
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return bool(grams) and len(set(grams)) / len(grams) < 0.7


def compare(sft: dict[str, dict[str, Any]], dpo: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ids = sorted(sft)
    t_sft = {i: sft[i]["n_tokens"] >= MAX_NEW_TOKENS for i in ids}
    t_dpo = {i: dpo[i]["n_tokens"] >= MAX_NEW_TOKENS for i in ids}
    b = sum(t_sft[i] and not t_dpo[i] for i in ids)     # tronquée en SFT seulement
    c = sum(t_dpo[i] and not t_sft[i] for i in ids)     # tronquée en DPO seulement
    lengths = [(sft[i]["n_tokens"], dpo[i]["n_tokens"]) for i in ids]
    res: dict[str, Any] = {
        "n": len(ids),
        "tronquees": {"sft": sum(t_sft.values()), "dpo": sum(t_dpo.values()),
                      "les_deux": sum(t_sft[i] and t_dpo[i] for i in ids),
                      "sft_seulement": b, "dpo_seulement": c, "mcnemar_p": round(mcnemar_exact(b, c), 4)},
        "tronquees_en_boucle": {"sft": sum(t_sft[i] and boucle(sft[i]["prediction"]) for i in ids),
                                "dpo": sum(t_dpo[i] and boucle(dpo[i]["prediction"]) for i in ids)},
        "longueur_mediane": {"sft": statistics.median(x for x, _ in lengths),
                             "dpo": statistics.median(y for _, y in lengths)},
        "part_dpo_plus_longue": round(sum(y > x for x, y in lengths) / len(lengths), 4),
    }
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sft", required=True, help="predictions.jsonl de l'évaluation SFT (pénalité 1.1)")
    parser.add_argument("--dpo", required=True, help="predictions.jsonl de l'évaluation DPO")
    parser.add_argument("--rapport", default="comparaison_sft_dpo.json", help="nom du rapport dans reports/")
    parser.add_argument("--sans-grille", action="store_true", help="comparaison seule, sans grille de relecture")
    args = parser.parse_args()

    sft, dpo = load(args.sft), load(args.dpo)
    for cat in CATEGORIES:
        if set(sft[cat]) != set(dpo[cat]):
            raise SystemExit(f"[{cat}] les deux évaluations ne portent pas sur les mêmes exemples")

    resultats: dict[str, Any] = {"sft": args.sft, "dpo": args.dpo, "max_new_tokens": MAX_NEW_TOKENS}
    for cat in CATEGORIES:
        resultats[cat] = compare(sft[cat], dpo[cat])
    clin = sorted(sft["clinique"])
    resultats["clinique"]["reponses_en_anglais"] = {
        k: sum(langue(src[i]["prediction"]) == "en" for i in clin) for k, src in (("sft", sft["clinique"]),
                                                                              ("dpo", dpo["clinique"]))}

    print(json.dumps(resultats, ensure_ascii=False, indent=2))
    # Contrôle du fichier SFT : doit redonner 17,1 % de cas cliniques tronqués (run à pénalité 1.1)
    r = resultats["clinique"]
    print(f"\nCas cliniques tronqués : SFT {r['tronquees']['sft'] / r['n']:.1%} "
          f"(attendu 17,1 %) — DPO {r['tronquees']['dpo'] / r['n']:.1%}")
    for i in [i for i in clin if dpo["clinique"][i]["n_tokens"] >= MAX_NEW_TOKENS
              and sft["clinique"][i]["n_tokens"] < MAX_NEW_TOKENS][:2]:
        print("-" * 80 + f"\n[tronquée en DPO seulement] {i} — fin de la réponse :\n"
              f"…{dpo['clinique'][i]['prediction'][-400:]}")
    report = ROOT / "reports" / args.rapport
    report.write_text(json.dumps(resultats, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.sans_grille:
        print(f"\nRapport : {report}")
        return

    # ---- Grille de relecture en aveugle (jamais écrasée : elle porte des décisions humaines)
    if (ROOT / "data" / "relecture" / "grille_relecture.csv").exists():
        raise SystemExit("data/relecture/grille_relecture.csv existe déjà : relancer avec --sans-grille")
    data_cfg = yaml.safe_load((ROOT / "configs" / "dpo.yaml").read_text(encoding="utf-8"))["dataset"]
    eval_cfg = yaml.safe_load((ROOT / "configs" / "eval.yaml").read_text(encoding="utf-8"))["dataset"]
    from datasets import load_dataset
    records = {r["id"]: r for r in load_dataset(eval_cfg["repo_id"], eval_cfg["config"],
                                                split=eval_cfg["clinique_split"],
                                                revision=data_cfg["revision"])}
    rng = random.Random(GRAINE)
    tires = rng.sample(clin, N_RELECTURE)
    out_dir = ROOT / "data" / "relecture"
    out_dir.mkdir(parents=True, exist_ok=True)
    cle: dict[str, dict[str, str]] = {}
    with (out_dir / "grille_relecture.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["id", "question", "reference", "reponse_A", "reponse_B",
                         "verdict_A", "verdict_B", "commentaire"])
        for i in tires:
            ordre = ["sft", "dpo"]
            rng.shuffle(ordre)
            cle[i] = {"A": ordre[0], "B": ordre[1]}
            src = {"sft": sft["clinique"][i], "dpo": dpo["clinique"][i]}
            question = next(m["content"] for m in records[i]["messages"] if m["role"] == "user")
            writer.writerow([i, question, src["sft"]["reference"], src[ordre[0]]["prediction"],
                             src[ordre[1]]["prediction"], "", "", ""])
    (out_dir / "cle_relecture.json").write_text(json.dumps(cle, indent=2), encoding="utf-8")
    print(f"\nRapport : {report}\nGrille : {out_dir / 'grille_relecture.csv'} "
          f"({N_RELECTURE} cas ; verdicts : juste / partiel / faux / dangereux)\n"
          f"Clé : {out_dir / 'cle_relecture.json'} — à n'ouvrir qu'après la relecture")


if __name__ == "__main__":
    main()
