"""Tests de la passerelle, sans réseau ni GPU : vLLM est remplacé par un faux serveur
(httpx.MockTransport) qui enregistre les requêtes reçues et renvoie des réponses contrôlées."""
import hashlib
import json
import re
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from src.api.app import creer_app
from src.api.config import Secrets, charger_config
from src.api.journal import Journal

RACINE = Path(__file__).resolve().parents[1]
CONFIG = charger_config(RACINE / "configs" / "serving.yaml")
CLE = "cle-de-test"
EN_TETES = {CONFIG.passerelle.en_tete_cle: CLE}
SFT = CONFIG.passerelle.modele_par_defaut


class FauxVllm:
    """Faux serveur vLLM : réponses réglables, requêtes conservées pour inspection."""

    def __init__(self):
        self.texte = "Les signes sont une dyspnée et une douleur thoracique."
        self.raison_fin = "stop"
        self.texte_priorite = "Réponse : B. Tri 2"
        self.statut_http = 200
        self.panne: Exception | None = None
        self.requetes: list[dict] = []

    def __call__(self, requete: httpx.Request) -> httpx.Response:
        if self.panne:
            raise self.panne
        if requete.method == "GET" and requete.url.path == "/v1/models":
            if self.statut_http != 200:
                return httpx.Response(self.statut_http)
            return httpx.Response(200, json={"data": [{"id": nom} for nom in CONFIG.modeles]})
        corps = json.loads(requete.content)
        self.requetes.append(corps)
        if self.statut_http != 200:
            return httpx.Response(self.statut_http, text="erreur simulée")
        priorite = corps["messages"][-1]["content"].startswith("Cas clinique :")
        return httpx.Response(200, json={
            "choices": [{"message": {"content": self.texte_priorite if priorite else self.texte},
                         "finish_reason": "stop" if priorite else self.raison_fin}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30},
        })

    def requete(self, priorite: bool) -> dict:
        return next(r for r in self.requetes
                    if r["messages"][-1]["content"].startswith("Cas clinique :") == priorite)


@pytest.fixture
def vllm() -> FauxVllm:
    return FauxVllm()


@pytest.fixture
def journal(tmp_path) -> Journal:
    return Journal(tmp_path / "journal.sqlite", CONFIG.journal.conservation_jours)


@pytest.fixture
def client(vllm, journal):
    secrets = Secrets(vllm_url="http://vllm.test", vllm_api_key="cle-vllm",
                      empreintes_cles=frozenset({hashlib.sha256(CLE.encode()).hexdigest()}))
    app = creer_app(config=CONFIG, secrets=secrets, journal=journal,
                    transport=httpx.MockTransport(vllm))
    with TestClient(app) as c:          # « with » : exécute le cycle de vie (création du client vLLM)
        yield c


def poser(client, **corps):
    return client.post("/v1/reponses", json={"question": "Signes d'une embolie pulmonaire ?", **corps},
                       headers=EN_TETES)


# ----------------------------------------------------------------------- authentification
@pytest.mark.parametrize("en_tetes", [{}, {CONFIG.passerelle.en_tete_cle: "mauvaise-cle"}])
def test_cle_absente_ou_invalide_refusee(client, vllm, en_tetes):
    r = client.post("/v1/reponses", json={"question": "test"}, headers=en_tetes)
    assert r.status_code == 401
    assert vllm.requetes == []                       # vLLM n'est jamais appelé


# ----------------------------------------------------------------------- contrôles d'entrée
@pytest.mark.parametrize("corps", [
    {"question": "   "},                                           # vide après nettoyage
    {"question": "x" * (CONFIG.passerelle.question_max_caracteres + 1)},
    {"question": "test", "modele": "gpt-4"},                       # modèle non servi
    {"question": "test", "modèle": "chsa-dpo"},                    # champ inconnu (faute de frappe)
])
def test_requete_invalide_refusee(client, vllm, corps):
    r = client.post("/v1/reponses", json=corps, headers=EN_TETES)
    assert r.status_code == 422
    assert vllm.requetes == []


# ----------------------------------------------------------------------- cas nominal
def test_reponse_complete_avec_priorite(client, vllm, journal):
    r = poser(client)
    assert r.status_code == 200
    donnees = r.json()
    assert donnees["modele"] == SFT                                # SFT par défaut
    assert donnees["revision"] == CONFIG.modeles[SFT].revision
    assert donnees["reponse"] == vllm.texte
    assert donnees["complete"] is True
    assert donnees["avertissement"] == CONFIG.passerelle.avertissement
    assert donnees["priorite"] == {"niveau": "Tri 2", "echelle": CONFIG.priorite.echelle,
                                   "statut": "non_validee", "format_valide": True}

    ligne = journal.lire(donnees["id_interaction"])               # traçabilité
    assert ligne["statut"] == "ok" and ligne["reponse"] == vllm.texte
    assert ligne["cle"] == hashlib.sha256(CLE.encode()).hexdigest()[:8]
    assert CLE not in json.dumps(ligne)                            # la clé n'est jamais journalisée
    assert ligne["priorite_brute"] == vllm.texte_priorite
    assert set(json.loads(ligne["parametres"])) == {"reponse", "priorite"}


def test_parametres_envoyes_identiques_a_l_evaluation(client, vllm):
    poser(client)
    gen, p = CONFIG.generation, CONFIG.priorite
    principale, priorite = vllm.requete(priorite=False), vllm.requete(priorite=True)
    for requete in (principale, priorite):
        assert requete["messages"][0] == {"role": "system", "content": gen.consigne_systeme}
        assert requete["temperature"] == 0.0
        assert requete["stop_token_ids"] == gen.stop_token_ids
        assert requete["chat_template_kwargs"] == {"enable_thinking": False}
    assert principale["messages"][1]["content"] == "Signes d'une embolie pulmonaire ?"
    assert (principale["max_tokens"], principale["repetition_penalty"]) == (gen.max_tokens, gen.repetition_penalty)
    assert (priorite["max_tokens"], priorite["repetition_penalty"]) == (p.max_tokens, p.repetition_penalty)


def test_modele_dpo_sur_demande_explicite(client, vllm):
    r = poser(client, modele="chsa-dpo")
    assert r.status_code == 200
    assert r.json()["revision"] == CONFIG.modeles["chsa-dpo"].revision
    assert {req["model"] for req in vllm.requetes} == {"chsa-dpo"}


def test_reponse_tronquee_signalee(client, vllm, journal):
    vllm.raison_fin = "length"
    donnees = poser(client).json()
    assert donnees["complete"] is False and donnees["raison_fin"] == "length"
    assert journal.lire(donnees["id_interaction"])["complete"] == 0


@pytest.mark.parametrize("sortie", ["Tri 2", "Réponses : A, B", "Je ne sais pas."])
def test_priorite_hors_format_non_devinee(client, vllm, journal, sortie):
    vllm.texte_priorite = sortie
    donnees = poser(client).json()
    assert donnees["priorite"]["niveau"] is None
    assert donnees["priorite"]["format_valide"] is False
    assert donnees["priorite"]["statut"] == "non_validee"
    assert journal.lire(donnees["id_interaction"])["priorite_brute"] == sortie


def test_sans_priorite_un_seul_appel(client, vllm):
    donnees = poser(client, inclure_priorite=False).json()
    assert donnees["priorite"] is None
    assert len(vllm.requetes) == 1


# ----------------------------------------------------------------------- pannes de vLLM
def _id_dans_le_message(reponse) -> str:
    return re.search(r"interaction ([0-9a-f-]{36})", reponse.json()["detail"]).group(1)


@pytest.mark.parametrize("panne, statut_http, code", [
    (httpx.ConnectError("connexion refusée"), 200, 503),   # pod GPU éteint
    (None, 500, 503),                                      # erreur serveur vLLM
    (None, 400, 502),                                      # réponse inattendue
])
def test_panne_vllm_journalisee(client, vllm, journal, panne, statut_http, code):
    vllm.panne, vllm.statut_http = panne, statut_http
    r = poser(client)
    assert r.status_code == code
    ligne = journal.lire(_id_dans_le_message(r))          # l'erreur renvoie à sa ligne de journal
    assert ligne["statut"] == "erreur" and ligne["erreur"]


# ----------------------------------------------------------------------- santé
def test_sante_ok(client):
    donnees = client.get("/sante").json()
    assert (donnees["statut"], donnees["vllm"]) == ("ok", "ok")
    assert [m["nom_servi"] for m in donnees["modeles"] if m["par_defaut"]] == [SFT]


def test_sante_degradee_si_vllm_indisponible(client, vllm):
    vllm.panne = httpx.ConnectError("connexion refusée")
    r = client.get("/sante")
    assert r.status_code == 200                          # la passerelle répond même GPU éteint
    assert (r.json()["statut"], r.json()["vllm"]) == ("degrade", "indisponible")
