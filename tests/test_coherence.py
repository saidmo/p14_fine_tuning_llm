"""La passerelle recopie quelques éléments de l'entraînement et de l'évaluation pour rester
indépendante (image légère, sans torch ni datasets). Ces tests garantissent que les copies ne
divergent pas des originaux : le modèle servi doit voir exactement ce qu'il a vu en SFT et être
lu exactement comme il a été évalué.

Les fonctions originales sont lues dans le SOURCE (module ast) et non importées : importer
src/data/normalize.py ou src/evaluation/evaluate.py chargerait datasets et torch.
"""
import ast
import re
from pathlib import Path

import pytest
import yaml

from src.api import vllm_client
from src.api.config import charger_config

RACINE = Path(__file__).resolve().parents[1]
SERVING = charger_config(RACINE / "configs" / "serving.yaml")


def _yaml(nom: str) -> dict:
    return yaml.safe_load((RACINE / "configs" / nom).read_text(encoding="utf-8"))


def _noeuds(fichier: str, noms: set[str]) -> dict[str, ast.AST]:
    """Définitions de premier niveau (fonctions et affectations) portant ces noms."""
    arbre = ast.parse((RACINE / fichier).read_text(encoding="utf-8"))
    trouves = {}
    for noeud in arbre.body:
        if isinstance(noeud, ast.FunctionDef) and noeud.name in noms:
            trouves[noeud.name] = noeud
        elif isinstance(noeud, ast.Assign):
            for cible in noeud.targets:
                if isinstance(cible, ast.Name) and cible.id in noms:
                    trouves[cible.id] = noeud
    assert set(trouves) == noms, f"introuvables dans {fichier} : {noms - set(trouves)}"
    return trouves


def _executer(fichier: str, noms: set[str]) -> dict:
    """Exécute seulement ces définitions, dans l'ordre du fichier."""
    noeuds = sorted(_noeuds(fichier, noms).values(), key=lambda n: n.lineno)
    espace = {"re": re}
    exec(compile(ast.Module(body=noeuds, type_ignores=[]), fichier, "exec"), espace)
    return espace


# ----------------------------------------------------------------------- prompt
def test_consigne_systeme_identique_a_l_entrainement():
    assert SERVING.generation.consigne_systeme == _yaml("format.yaml")["system_prompt"]


def test_mode_reflexion_identique_a_l_entrainement():
    assert SERVING.generation.chat_template_kwargs == _yaml("sft.yaml")["model"]["chat_template_kwargs"]


def test_parametres_identiques_a_l_evaluation():
    evaluation = _yaml("eval.yaml")["generation"]
    assert SERVING.generation.max_tokens == evaluation["max_new_tokens_ouvert"]
    assert SERVING.generation.repetition_penalty == evaluation["repetition_penalty_ouvert"]
    assert SERVING.priorite.max_tokens == evaluation["max_new_tokens_qcm"]
    assert SERVING.priorite.repetition_penalty == 1.0          # pas de pénalité sur les QCM
    assert SERVING.generation.temperature == 0.0               # greedy, comme à l'évaluation


# ----------------------------------------------------------------------- priorité (QCU)
def test_gabarit_qcu_identique_a_format_mcq():
    original = _executer("src/data/normalize.py", {"format_mcq", "CONSIGNES_QCM"})
    p = SERVING.priorite
    assert p.consigne == original["CONSIGNES_QCM"]["mcq_single"]
    cas = "Homme de 70 ans, douleur thoracique depuis une heure, sueurs."
    assert (vllm_client.format_qcu(cas, p.question, p.options, p.consigne)
            == original["format_mcq"](p.question, p.options, cas, "mcq_single"))


def test_extraction_identique_a_l_evaluation():
    """Même code, au caractère près (hors commentaires) : ast.dump ignore la mise en forme."""
    noms = {"_LETTER", "_ANSWER", "_LEADING_LETTER", "parse_letters"}
    evaluation = _noeuds("src/evaluation/evaluate.py", noms)
    passerelle = _noeuds("src/api/vllm_client.py", noms)
    for nom in noms:
        assert ast.dump(passerelle[nom]) == ast.dump(evaluation[nom]), nom


# ----------------------------------------------------------------------- tokens d'arrêt
def test_tokens_d_arret():
    """Nécessite transformers et le réseau : ignoré dans la CI de la passerelle, lancé sur le poste."""
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        SERVING.vllm.modele.repo, revision=SERVING.vllm.modele.revision)
    assert SERVING.generation.stop_token_ids == tokenizer.convert_tokens_to_ids(
        ["<|im_end|>", "<|endoftext|>"])
