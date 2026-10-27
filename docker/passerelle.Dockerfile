# Passerelle FastAPI du POC CHSA (semaine 4) — construite et publiée par la CI.
# Image légère et PUBLIABLE : ni poids de modèle, ni secret (fournis par l'environnement
# au démarrage : VLLM_URL, VLLM_API_KEY, CHSA_API_KEYS).
#
# Construction, depuis la racine du dépôt :
#   docker build -f docker/passerelle.Dockerfile -t chsa-passerelle:dev .

# Même version mineure que l'environnement de test (.venv-api : Python 3.12).

# python:3.12-slim (linux/amd64) — épinglé le 2026-10-27
FROM python:3.12-slim@sha256:6d2b0577e6d1632f077331de8e4780535f54839d163ded7abb4b7dd37b5d37ac

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dépendances d'abord : cette couche n'est reconstruite que si le .lock change.
COPY requirements-api.lock .
RUN pip install -r requirements-api.lock

# Uniquement ce dont la passerelle a besoin : sa configuration et son code.
COPY configs/serving.yaml configs/serving.yaml
COPY src/__init__.py src/__init__.py
COPY src/api/ src/api/

# Utilisateur sans privilèges ; /data reçoit le volume du journal (journal.chemin).
RUN useradd --create-home --uid 1000 chsa \
    && mkdir -p /data \
    && chown chsa:chsa /data
USER chsa

EXPOSE 8080

# Vivacité de la passerelle seule : /sante répond 200 même si le pod GPU est éteint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/sante', timeout=4)"

CMD ["uvicorn", "src.api.app:creer_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
