"""Filtre devant vLLM, sur le pod GPU : seul point exposé à Internet (port public).

Pourquoi : la clé d'API de vLLM ne protège que les routes /v1/… Au premier déploiement, la route
/invocations (format SageMaker) générait du texte SANS clé, contournant la clé et la passerelle
(donc la journalisation) ; /metrics, /docs et des routes d'administration étaient aussi ouvertes.

Principe : vLLM n'écoute plus que sur le pod (127.0.0.1) ; ce filtre applique
  1. une LISTE BLANCHE (méthode, chemin) : tout le reste répond 404, y compris les routes qu'une
     future version de vLLM ajouterait ;
  2. la vérification de la clé, à temps constant, avant toute transmission.
vLLM garde sa propre clé (défense en profondeur) : l'en-tête Authorization lui est transmis.
Le chemin transmis est celui de la liste blanche, jamais celui reçu.

Dépendances : starlette, httpx et uvicorn, déjà présents dans l'image vLLM (rien à installer).
Lancé par src/serving/demarrer_vllm.py.
"""
import hmac
from contextlib import asynccontextmanager

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

# (méthode, chemin) → clé exigée. Ajouter une route est une décision de sécurité, testée.
ROUTES_AUTORISEES: dict[tuple[str, str], bool] = {
    ("POST", "/v1/chat/completions"): True,     # génération (passerelle, validation)
    ("GET", "/v1/models"): True,                # modèles servis (santé de la passerelle)
    ("GET", "/version"): True,                  # version de vLLM (rapport de validation)
    ("POST", "/tokenize"): True,                # identité du prompt avec l'entraînement
    ("GET", "/health"): False,                  # surveillance : ne révèle rien
}
EN_TETES_TRANSMIS = {"authorization", "content-type"}
METHODES = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]


def creer_filtre(cle: str, cible: str, transport: httpx.AsyncBaseTransport | None = None,
                 delai_secondes: float = 300) -> Starlette:
    """`cible` : URL interne de vLLM ; `transport` : faux vLLM dans les tests."""
    if not cle:
        raise ValueError("clé d'API vide : le filtre refuse de démarrer")
    attendu = f"Bearer {cle}".encode()

    @asynccontextmanager
    async def cycle_de_vie(app: Starlette):
        app.state.client = httpx.AsyncClient(base_url=cible, timeout=delai_secondes, transport=transport)
        yield
        await app.state.client.aclose()

    async def filtrer(requete: Request) -> Response:
        route = (requete.method, requete.url.path)
        if route not in ROUTES_AUTORISEES:
            return JSONResponse({"detail": "route non disponible"}, status_code=404)
        if ROUTES_AUTORISEES[route]:
            recu = requete.headers.get("authorization", "").encode()
            if not hmac.compare_digest(recu, attendu):
                return JSONResponse({"detail": "clé d'API absente ou invalide"}, status_code=401)
        en_tetes = {k: v for k, v in requete.headers.items() if k.lower() in EN_TETES_TRANSMIS}
        try:
            reponse = await requete.app.state.client.request(*route, content=await requete.body(),
                                                              headers=en_tetes)
        except httpx.HTTPError:
            return JSONResponse({"detail": "vLLM indisponible"}, status_code=503)
        return Response(reponse.content, status_code=reponse.status_code,
                        media_type=reponse.headers.get("content-type"))

    return Starlette(routes=[Route("/{chemin:path}", filtrer, methods=METHODES)],
                     lifespan=cycle_de_vie)
