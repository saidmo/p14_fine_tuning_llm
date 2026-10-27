"""Diagnostic des désaccords de lettres entre vLLM et transformers (QCM du volet cohérence).

Hypothèse testée : les désaccords sont des bascules entre tokens presque à égalité, dues aux
différences numériques bf16 entre les deux moteurs (et au remplissage des lots de l'évaluation),
et non un défaut du déploiement.

Méthode : le désaccord à expliquer oppose A (réponse de vLLM pendant la validation) et B (réponse
de transformers). On trouve la première position où A et B divergent (tokens a et b). Chaque QCM est
rejoué sur vLLM (run C) avec les mêmes paramètres et les log-probabilités des 5 meilleurs tokens ;
si C a le même début que A jusqu'à cette position, on mesure |logprob(a) − logprob(b)| dans C,
quel que soit le token choisi par C. Non mesurable (début différent, a ou b hors top 5) = compté
comme NON quasi-égalité. Les tokens d'arrêt sont retirés avant toute comparaison.

Critère fixé AVANT la mesure : quasi-égalité si écart ≤ SEUIL_QUASI_EGALITE ; hypothèse confirmée
si au moins PART_MIN_CONFIRMATION des désaccords de lettres sont des quasi-égalités.

À lancer depuis la racine du dépôt, dans .venv (datasets, transformers), vLLM en marche :
    python -m src.serving.diagnostic_accord --rapport reports/validation_deploiement_<date>.json
Variables : VLLM_URL, VLLM_API_KEY.
"""
import argparse
import asyncio
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

import httpx

from src.api.config import ConfigServing, charger_config
from src.serving.valider_deploiement import en_parallele, lire_predictions, lire_questions

SEUIL_QUASI_EGALITE = 0.25    # nats : ordre de grandeur des écarts numériques bf16 (fixé a priori)
PART_MIN_CONFIRMATION = 0.90


def id_token(token: str) -> int:
    return int(token.split(":", 1)[1])            # « token_id:1234 » (return_tokens_as_token_ids)


async def sequence_vllm(client: httpx.AsyncClient, config: ConfigServing, modele: str,
                        question: str) -> list[tuple[int, dict[int, float]]]:
    """Tokens générés et log-probabilités des 5 meilleurs candidats à chaque position.
    Paramètres identiques au volet cohérence (ceux de l'évaluation des QCM)."""
    gen, p = config.generation, config.priorite
    r = await client.post("/v1/chat/completions", json={
        "model": modele,
        "messages": [{"role": "system", "content": gen.consigne_systeme},
                     {"role": "user", "content": question}],
        "temperature": gen.temperature, "max_tokens": p.max_tokens,
        "repetition_penalty": p.repetition_penalty, "stop_token_ids": gen.stop_token_ids,
        "chat_template_kwargs": gen.chat_template_kwargs,
        "logprobs": True, "top_logprobs": 5, "return_tokens_as_token_ids": True})
    r.raise_for_status()
    contenu = r.json()["choices"][0]["logprobs"]["content"]
    return [(id_token(t["token"]), {id_token(a["token"]): a["logprob"] for a in t["top_logprobs"]})
            for t in contenu]


def premiere_divergence(ids_a: list[int], ids_b: list[int]) -> int | None:
    """Première position où deux séquences de tokens diffèrent (None si identiques)."""
    for i, (x, y) in enumerate(zip(ids_a, ids_b)):
        if x != y:
            return i
    return None if len(ids_a) == len(ids_b) else min(len(ids_a), len(ids_b))


def ecart_a_la_divergence(seq_c: list[tuple[int, dict[int, float]]], ids_a: list[int],
                          ids_b: list[int], i: int) -> dict:
    """Écart, dans la distribution du run C, entre les deux tokens a et b qui ont divergé."""
    if i >= len(ids_a) or i >= len(ids_b):
        return {"mesurable": False, "cause": "une séquence s'arrête plus tôt"}
    if i >= len(seq_c) or [t for t, _ in seq_c[:i]] != ids_a[:i]:
        return {"mesurable": False, "cause": "début du run C différent de A"}
    top, a, b = seq_c[i][1], ids_a[i], ids_b[i]
    if a not in top or b not in top:
        return {"mesurable": False, "cause": "a ou b hors top 5"}
    return {"mesurable": True, "ecart": abs(top[a] - top[b]), "c_choisit": "a" if seq_c[i][0] == a else "b"}


def ecarts_top2(seq) -> list[float]:
    """Écart entre les deux meilleurs candidats à chaque position : confiance typique du modèle."""
    return [sorted(top.values(), reverse=True)[0] - sorted(top.values(), reverse=True)[1]
            for _, top in seq if len(top) > 1]


async def diagnostiquer(args, config: ConfigServing) -> Path:
    from transformers import AutoTokenizer
    rapport = json.loads(Path(args.rapport).read_text(encoding="utf-8"))
    with open(rapport["details"], encoding="utf-8") as f:
        qcm = [d for d in map(json.loads, f) if d["volet"] == "coherence_qcm"]
    refs = {m: lire_predictions(r["fichier"]) for m, r in rapport["contexte"]["references"].items()}
    questions, _ = lire_questions()
    m = config.vllm.modele
    tokenizer = AutoTokenizer.from_pretrained(m.repo, revision=m.revision)

    async with httpx.AsyncClient(base_url=os.environ["VLLM_URL"], timeout=180,
                                 headers={"Authorization": f"Bearer {os.environ['VLLM_API_KEY']}"}) as client:
        sequences = await en_parallele([lambda d=d: sequence_vllm(client, config, d["modele"],
                                                                   questions[("test", d["id"])])
                                        for d in qcm], 16)

    arrets = set(config.generation.stop_token_ids)
    encoder = lambda texte: tokenizer((texte or "").strip(), add_special_tokens=False)["input_ids"]
    lignes, resultats = [], {}
    for modele in refs:
        desaccords, accords_top2, reproductibles, n = [], [], 0, 0
        for d, seq in zip(qcm, sequences):
            if d["modele"] != modele:
                continue
            n += 1
            seq = [(t, top) for t, top in seq if t not in arrets]          # sans token d'arrêt
            ids_a = encoder(d["prediction"])                               # vLLM, run de validation
            ids_b = encoder(refs[modele]["qcm"][d["id"]]["prediction"])    # transformers
            reproductibles += [t for t, _ in seq] == ids_a                 # C reproduit-il A ?
            if d["predites"] == d["predites_transformers"]:
                accords_top2 += ecarts_top2(seq)
                continue
            i = premiere_divergence(ids_a, ids_b)
            mesure = ({"mesurable": False, "cause": "séquences identiques (extraction ?)"} if i is None
                      else ecart_a_la_divergence(seq, ids_a, ids_b, i))
            desaccords.append(mesure)
            lignes.append({"modele": modele, "id": d["id"], "position": i,
                           "c_reproduit_a": [t for t, _ in seq] == ids_a, **mesure})
        ecarts = [x["ecart"] for x in desaccords if x["mesurable"]]
        quasi = sum(e <= SEUIL_QUASI_EGALITE for e in ecarts)
        resultats[modele] = {
            "qcm": n, "desaccords": len(desaccords),
            "reproductibles_vllm": reproductibles / n if n else 0.0,   # run C identique au run A
            "desaccords_mesurables": len(ecarts),
            "ecart_median_desaccords": statistics.median(ecarts) if ecarts else None,
            "ecart_max_desaccords": max(ecarts) if ecarts else None,
            "non_mesurables": {c: sum(x.get("cause") == c for x in desaccords)
                               for c in sorted({x["cause"] for x in desaccords if not x["mesurable"]})},
            "quasi_egalites": quasi,
            "part_quasi_egalites": quasi / len(desaccords) if desaccords else 1.0,
            "c_rejoint_transformers": sum(x.get("c_choisit") == "b" for x in desaccords),
            "ecart_top2_median_accords": statistics.median(accords_top2) if accords_top2 else None,
        }
    resultats["criteres"] = {"seuil_quasi_egalite_nats": SEUIL_QUASI_EGALITE,
                             "part_min_confirmation": PART_MIN_CONFIRMATION}
    resultats["verdicts"] = {f"hypothese_confirmee_{m}": resultats[m]["part_quasi_egalites"] >= PART_MIN_CONFIRMATION
                             for m in refs}
    resultats["rapport_validation"] = args.rapport

    horodatage = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    details = Path(rapport["details"]).parent / f"diagnostic_accord_{horodatage}.jsonl"
    details.write_text("".join(json.dumps(l, ensure_ascii=False) + "\n" for l in lignes), encoding="utf-8")
    resultats["details"] = str(details)
    chemin = Path("reports") / f"diagnostic_accord_{horodatage}.json"
    chemin.write_text(json.dumps(resultats, ensure_ascii=False, indent=2), encoding="utf-8")
    return chemin


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rapport", required=True, help="rapport de validation dont on diagnostique les QCM")
    args = parser.parse_args()
    chemin = asyncio.run(diagnostiquer(args, charger_config()))
    print(Path(chemin).read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
