"""Client de la passerelle vers le serveur vLLM (API compatible OpenAI).

Deux appels par question, envoyés en parallèle par l'application :
  - la réponse principale, avec un prompt IDENTIQUE à l'entraînement (consigne système + question) ;
  - le niveau de priorité, posé en QCU dans le gabarit exact du SFT (src/data/normalize.py,
    format_mcq), puis extrait avec la règle exacte de l'évaluation (src/evaluation/evaluate.py,
    parse_letters). Ces deux copies sont vérifiées contre les originaux par des tests de la CI :
    la passerelle reste ainsi indépendante du code d'entraînement.
"""
import re
import time
from dataclasses import dataclass

import httpx

from src.api.config import ConfigServing, Secrets

# --- copie de src/evaluation/evaluate.py (parse_letters), vérifiée par un test -------------
_LETTER = r"[A-E](?![A-Za-zà-ÿœæŒÆ])"     # une lettre seule : pas le « E » de « Elle »
_ANSWER = re.compile(rf"[Rr][ée]ponses?\s*:\s*({_LETTER}(?:\s*(?:,|;|et)\s*{_LETTER})*)")
_LEADING_LETTER = re.compile(r"^\s*([A-E])\s*[.)]")


def parse_letters(text: str) -> list[str]:
    """Lettres de la réponse : « Réponse : D. … », « Réponses : B, C, D », sinon « D. … » en tête."""
    match = _ANSWER.search(text)
    if match:
        return sorted(set(re.findall(r"[A-E]", match.group(1))))
    match = _LEADING_LETTER.match(text)
    return [match.group(1)] if match else []
# -------------------------------------------------------------------------------------------


def format_qcu(cas: str, question: str, options: dict[str, str], consigne: str) -> str:
    """Copie de src/data/normalize.py (format_mcq) pour un QCU avec cas clinique."""
    return "\n\n".join([f"Cas clinique :\n{cas}",
                        f"Question : {question}",
                        "\n".join(f"{lettre}. {texte}" for lettre, texte in options.items()),
                        consigne])


class VllmIndisponible(Exception):
    """vLLM injoignable, délai dépassé ou erreur serveur : la passerelle répond 503."""


class VllmErreur(Exception):
    """Réponse de vLLM inattendue (requête refusée, format inconnu) : la passerelle répond 502."""


@dataclass(frozen=True)
class Generation:
    texte: str
    raison_fin: str
    tokens_prompt: int
    tokens_reponse: int
    latence_ms: int
    parametres: dict          # envoyés à vLLM, journalisés pour la reproductibilité


@dataclass(frozen=True)
class NiveauPriorite:
    niveau: str | None
    format_valide: bool
    brute: str


class ClientVllm:
    def __init__(self, config: ConfigServing, secrets: Secrets,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.config = config
        self._http = httpx.AsyncClient(               # transport : faux serveur vLLM dans les tests
            base_url=secrets.vllm_url,
            headers={"Authorization": f"Bearer {secrets.vllm_api_key}"},
            timeout=config.passerelle.delai_vllm_secondes,
            transport=transport,
        )

    async def fermer(self) -> None:
        await self._http.aclose()

    async def _chat(self, modele: str, contenu: str, max_tokens: int,
                    repetition_penalty: float) -> Generation:
        gen = self.config.generation
        parametres = {
            "temperature": gen.temperature,
            "max_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "stop_token_ids": gen.stop_token_ids,
            "chat_template_kwargs": gen.chat_template_kwargs,
        }
        corps = {"model": modele,
                 "messages": [{"role": "system", "content": gen.consigne_systeme},
                              {"role": "user", "content": contenu}],
                 **parametres}
        debut = time.perf_counter()
        try:
            reponse = await self._http.post("/v1/chat/completions", json=corps)
        except httpx.HTTPError as erreur:              # connexion refusée, délai dépassé…
            raise VllmIndisponible(f"{type(erreur).__name__} : {erreur}") from erreur
        latence_ms = round((time.perf_counter() - debut) * 1000)

        if reponse.status_code >= 500:
            raise VllmIndisponible(f"vLLM a répondu {reponse.status_code}")
        if reponse.status_code != 200:
            raise VllmErreur(f"vLLM a répondu {reponse.status_code} : {reponse.text[:300]}")
        try:
            donnees = reponse.json()
            choix = donnees["choices"][0]
            return Generation(texte=(choix["message"]["content"] or "").strip(),
                              raison_fin=choix["finish_reason"],
                              tokens_prompt=donnees["usage"]["prompt_tokens"],
                              tokens_reponse=donnees["usage"]["completion_tokens"],
                              latence_ms=latence_ms,
                              parametres=parametres)
        except (ValueError, KeyError, IndexError, TypeError) as erreur:
            raise VllmErreur(f"réponse de vLLM illisible : {erreur}") from erreur

    async def repondre(self, modele: str, question: str) -> Generation:
        gen = self.config.generation
        return await self._chat(modele, question, gen.max_tokens, gen.repetition_penalty)

    async def prioriser(self, modele: str, question: str) -> tuple[NiveauPriorite, Generation]:
        p = self.config.priorite
        generation = await self._chat(modele, format_qcu(question, p.question, p.options, p.consigne),
                                      p.max_tokens, p.repetition_penalty)
        lettres = parse_letters(generation.texte)
        valide = len(lettres) == 1 and lettres[0] in p.options    # QCU : exactement une lettre connue
        niveau = NiveauPriorite(niveau=p.options[lettres[0]] if valide else None,
                                format_valide=valide, brute=generation.texte)
        return niveau, generation

    async def sante(self) -> bool:
        """True si vLLM répond et sert exactement les modèles attendus."""
        try:
            reponse = await self._http.get("/v1/models", timeout=5)
            servis = {m["id"] for m in reponse.json()["data"]} if reponse.status_code == 200 else set()
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return False
        return set(self.config.modeles) <= servis
