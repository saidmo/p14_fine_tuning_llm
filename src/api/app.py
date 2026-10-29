"""Passerelle FastAPI du POC CHSA — seul point d'entrée des utilisateurs.

Lancement (factory : la configuration et les secrets sont lus au démarrage, pas à l'import) :
    uvicorn src.api.app:creer_app --factory --host 0.0.0.0 --port 8080

Pour chaque question : authentification par clé, contrôle du modèle et de la longueur, réponse
principale et priorité demandées EN PARALLÈLE à vLLM, une ligne de journal (succès ou erreur),
réponse avec avertissement, révision servie et identifiant d'interaction.
"""
import asyncio
import hashlib
import hmac
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from fastapi.concurrency import run_in_threadpool

from src.api.config import ConfigServing, Secrets, charger_config
from src.api.journal import Interaction, Journal
from src.api.schemas import ModeleServi, Priorite, ReponseQuestion, RequeteQuestion, Sante
from src.api.vllm_client import ClientVllm, VllmErreur, VllmIndisponible


def creer_app(config: ConfigServing | None = None, secrets: Secrets | None = None,
              journal: Journal | None = None,
              transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    """Les paramètres permettent aux tests d'injecter un faux vLLM et un journal temporaire."""
    config = config or charger_config()
    secrets = secrets or Secrets.depuis_environnement()
    journal = journal or Journal(config.journal.chemin, config.journal.conservation_jours)

    @asynccontextmanager
    async def cycle_de_vie(app: FastAPI):
        app.state.client = ClientVllm(config, secrets, transport=transport)
        yield
        await app.state.client.fermer()

    app = FastAPI(title="POC CHSA — assistant médical (démonstration)",
                  description=config.passerelle.avertissement, lifespan=cycle_de_vie)


    # Déclarée à FastAPI : /docs affiche un bouton « Authorize » pour saisir la clé.
    # auto_error=False : l'absence de clé passe par notre propre 401, comme une clé invalide.
    schema_cle = APIKeyHeader(name=config.passerelle.en_tete_cle, auto_error=False,
                              description="Clé d'accès personnelle (seule son empreinte est conservée).")

    def authentifier(cle: str | None = Security(schema_cle)) -> str:
        """Renvoie l'identifiant journalisé de la clé : 8 caractères de son empreinte."""
        empreinte = hashlib.sha256((cle or "").encode()).hexdigest()
        # comparaison à temps constant : la durée ne renseigne pas sur la proximité d'une clé valide
        if not cle or not any(hmac.compare_digest(empreinte, e) for e in secrets.empreintes_cles):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "clé d'API absente ou invalide")
        return empreinte[:8]
    
    @app.post("/v1/reponses", response_model=ReponseQuestion)
    async def repondre(corps: RequeteQuestion, requete: Request,
                       cle: str = Depends(authentifier)) -> ReponseQuestion:
        nom = corps.modele or config.passerelle.modele_par_defaut
        if nom not in config.modeles:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT,
                                f"modèle inconnu : {nom!r} (autorisés : {sorted(config.modeles)})")
        if len(corps.question) > config.passerelle.question_max_caracteres:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT,
                                f"question trop longue : {len(corps.question)} caractères "
                                f"(maximum {config.passerelle.question_max_caracteres})")

        client: ClientVllm = requete.app.state.client
        interaction = Interaction(id_interaction=str(uuid.uuid4()), cle=cle, modele=nom,
                                  revision=config.modeles[nom].revision, parametres={},
                                  question=corps.question)
        debut = time.perf_counter()
        try:
            appels = [client.repondre(nom, corps.question)]
            if corps.inclure_priorite:
                appels.append(client.prioriser(nom, corps.question))
            resultats = await asyncio.gather(*appels)
        except (VllmIndisponible, VllmErreur) as erreur:
            interaction.statut, interaction.erreur = "erreur", str(erreur)
            interaction.latence_ms = round((time.perf_counter() - debut) * 1000)
            await run_in_threadpool(journal.ecrire, interaction)
            code = (status.HTTP_503_SERVICE_UNAVAILABLE if isinstance(erreur, VllmIndisponible)
                    else status.HTTP_502_BAD_GATEWAY)
            raise HTTPException(code, f"modèle indisponible (interaction {interaction.id_interaction})")

        generation = resultats[0]
        interaction.parametres = {"reponse": generation.parametres}
        interaction.reponse = generation.texte
        interaction.complete = generation.raison_fin != "length"
        interaction.raison_fin = generation.raison_fin
        interaction.tokens_prompt = generation.tokens_prompt
        interaction.tokens_reponse = generation.tokens_reponse
        priorite = None
        if corps.inclure_priorite:
            niveau, generation_priorite = resultats[1]
            interaction.parametres["priorite"] = generation_priorite.parametres
            interaction.priorite_niveau = niveau.niveau
            interaction.priorite_format_valide = niveau.format_valide
            interaction.priorite_brute = niveau.brute
            priorite = Priorite(niveau=niveau.niveau, echelle=config.priorite.echelle,
                                format_valide=niveau.format_valide)
        interaction.latence_ms = round((time.perf_counter() - debut) * 1000)
        await run_in_threadpool(journal.ecrire, interaction)

        return ReponseQuestion(
            id_interaction=interaction.id_interaction, modele=nom, revision=interaction.revision,
            reponse=generation.texte, complete=interaction.complete,
            raison_fin=generation.raison_fin, priorite=priorite,
            avertissement=config.passerelle.avertissement, latence_ms=interaction.latence_ms,
            tokens_prompt=generation.tokens_prompt, tokens_reponse=generation.tokens_reponse)

    @app.get("/sante", response_model=Sante)
    async def sante(requete: Request) -> Sante:
        """Sans authentification : sert à la surveillance ; ne révèle que les révisions servies."""
        vllm_ok = await requete.app.state.client.sante()
        return Sante(statut="ok" if vllm_ok else "degrade",
                     vllm="ok" if vllm_ok else "indisponible",
                     modeles=[ModeleServi(nom_servi=m.nom_servi, repo=m.repo, revision=m.revision,
                                          par_defaut=m.nom_servi == config.passerelle.modele_par_defaut)
                              for m in config.modeles.values()])

    return app
