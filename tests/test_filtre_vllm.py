"""Filtre devant vLLM : liste blanche et clé, testés sans GPU (faux vLLM, httpx.MockTransport)."""
import httpx
import pytest
from starlette.testclient import TestClient

from src.serving.filtre_vllm import ROUTES_AUTORISEES, creer_filtre

CLE = "cle-vllm-test"
AUTORISE = {"Authorization": f"Bearer {CLE}"}


class FauxVllm:
    def __init__(self):
        self.recues: list[httpx.Request] = []
        self.panne = False

    def __call__(self, requete: httpx.Request) -> httpx.Response:
        if self.panne:
            raise httpx.ConnectError("vLLM arrêté")
        self.recues.append(requete)
        return httpx.Response(200, json={"chemin": requete.url.path})


@pytest.fixture
def vllm() -> FauxVllm:
    return FauxVllm()


@pytest.fixture
def client(vllm):
    with TestClient(creer_filtre(CLE, "http://127.0.0.1:8001", transport=httpx.MockTransport(vllm))) as c:
        yield c


def test_cle_vide_refusee():
    with pytest.raises(ValueError):
        creer_filtre("", "http://127.0.0.1:8001")


@pytest.mark.parametrize("methode, chemin", sorted(ROUTES_AUTORISEES))
def test_routes_autorisees_transmises_avec_la_cle(client, vllm, methode, chemin):
    r = client.request(methode, chemin, headers=AUTORISE, json={} if methode == "POST" else None)
    assert r.status_code == 200 and r.json() == {"chemin": chemin}
    assert vllm.recues[-1].headers["authorization"] == AUTORISE["Authorization"]   # défense en profondeur


@pytest.mark.parametrize("en_tetes", [{}, {"Authorization": "Bearer mauvaise"}, {"Authorization": CLE}])
def test_cle_absente_ou_invalide_refusee(client, vllm, en_tetes):
    r = client.post("/v1/chat/completions", headers=en_tetes, json={})
    assert r.status_code == 401
    assert vllm.recues == []                                       # rien n'atteint vLLM


def test_health_sans_cle(client, vllm):
    assert client.get("/health").status_code == 200


@pytest.mark.parametrize("methode, chemin", [
    ("POST", "/invocations"),            # contournement constaté au premier déploiement
    ("GET", "/metrics"), ("GET", "/docs"), ("GET", "/openapi.json"),
    ("POST", "/scale_elastic_ep"), ("POST", "/generative_scoring"),
    ("POST", "/v1/completions"), ("POST", "/v1/responses"),        # /v1 mais non utilisées
    ("GET", "/v1/chat/completions"),     # bonne route, mauvaise méthode
    ("GET", "/v1/models/"),              # variante du chemin
    ("POST", "/v1/../invocations"),      # tentative de remontée de chemin
])
def test_routes_hors_liste_fermees_meme_avec_la_cle(client, vllm, methode, chemin):
    r = client.request(methode, chemin, headers=AUTORISE)
    assert r.status_code == 404
    assert vllm.recues == []


def test_vllm_indisponible(client, vllm):
    vllm.panne = True
    assert client.get("/v1/models", headers=AUTORISE).status_code == 503
