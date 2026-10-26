"""Configuration du déploiement : chargement et validation de configs/serving.yaml.

Fermé par défaut, comme le reste du projet : une configuration incohérente arrête le démarrage
au lieu de produire un service qui répond avec le mauvais modèle ou la mauvaise révision.
Les secrets ne sont jamais dans le fichier : ils sont lus dans les variables d'environnement.
"""
import os
import re
from functools import cached_property
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CHEMIN_PAR_DEFAUT = "configs/serving.yaml"
SHA_REVISION = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")
LETTRES_QCM = "ABCDE"           # lettres vues par le SFT (src/data/normalize.py)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)   # clé inconnue = faute de frappe


class ModeleHub(_Strict):
    nom_servi: str
    repo: str
    revision: str

    @field_validator("revision")
    @classmethod
    def revision_epinglee(cls, valeur: str) -> str:
        if not SHA_REVISION.fullmatch(valeur):
            raise ValueError(f"révision non épinglée (SHA de 40 caractères attendu) : {valeur!r}")
        return valeur


class Adaptateur(ModeleHub):
    rang: int = Field(gt=0)


class ConfigVllm(_Strict):
    image: str
    port: int
    dtype: str
    max_model_len: int = Field(gt=0)
    gpu_memory_utilization: float = Field(gt=0, le=1)
    dossier_modeles: str
    modele: ModeleHub
    adaptateurs: list[Adaptateur] = []

    def verifier_image(self) -> None:
        """À appeler avant tout déploiement du pod GPU (CI, script de démarrage).

        La passerelle n'utilise pas l'image vLLM : ce contrôle n'empêche donc pas de la tester.
        """
        nom, _, version = self.image.partition(":")
        if not version or version == "latest" or "<" in version:
            raise ValueError(f"image vLLM non épinglée : {self.image!r}")


class ConfigPasserelle(_Strict):
    modele_par_defaut: str
    modeles_autorises: list[str]
    question_max_caracteres: int = Field(gt=0)
    delai_vllm_secondes: float = Field(gt=0)
    en_tete_cle: str
    avertissement: str


class ConfigGeneration(_Strict):
    consigne_systeme: str
    chat_template_kwargs: dict[str, bool]
    temperature: float = Field(ge=0)
    max_tokens: int = Field(gt=0)
    repetition_penalty: float = Field(gt=0)
    stop_token_ids: list[int] = Field(min_length=1)


class ConfigPriorite(_Strict):
    statut: str
    echelle: str
    question: str
    options: dict[str, str]
    consigne: str
    max_tokens: int = Field(gt=0)
    repetition_penalty: float = Field(gt=0)

    @field_validator("options")
    @classmethod
    def lettres_connues(cls, options: dict[str, str]) -> dict[str, str]:
        lettres = "".join(options)
        if len(options) < 2 or lettres != LETTRES_QCM[:len(options)]:
            raise ValueError(f"options attendues A, B, C… dans l'ordre (au plus {LETTRES_QCM}) : {lettres!r}")
        return options


class ConfigJournal(_Strict):
    chemin: str
    conservation_jours: int = Field(gt=0)


class ConfigServing(_Strict):
    vllm: ConfigVllm
    passerelle: ConfigPasserelle
    generation: ConfigGeneration
    priorite: ConfigPriorite
    journal: ConfigJournal
    validation: dict          # lu par le script de validation du déploiement, pas par la passerelle

    @model_validator(mode="after")
    def modeles_coherents(self) -> "ConfigServing":
        servis = [self.vllm.modele.nom_servi] + [a.nom_servi for a in self.vllm.adaptateurs]
        if len(set(servis)) != len(servis):
            raise ValueError(f"noms servis en double : {servis}")
        if set(self.passerelle.modeles_autorises) != set(servis):
            raise ValueError(f"modeles_autorises {self.passerelle.modeles_autorises} "
                             f"≠ modèles servis par vLLM {servis}")
        if self.passerelle.modele_par_defaut != self.vllm.modele.nom_servi:
            raise ValueError("le modèle par défaut doit être le modèle de base de vLLM "
                             f"({self.vllm.modele.nom_servi}), pas un adaptateur")
        return self

    @cached_property
    def modeles(self) -> dict[str, ModeleHub]:
        """Modèles servis, indexés par leur nom servi."""
        return {m.nom_servi: m for m in [self.vllm.modele, *self.vllm.adaptateurs]}


def charger_config(chemin: str | Path | None = None) -> ConfigServing:
    chemin = Path(chemin or os.environ.get("CHSA_SERVING_CONFIG", CHEMIN_PAR_DEFAUT))
    with chemin.open(encoding="utf-8") as f:
        return ConfigServing.model_validate(yaml.safe_load(f))


class Secrets(_Strict):
    """Secrets de la passerelle, lus dans l'environnement (secrets RunPod)."""

    vllm_url: str
    vllm_api_key: str = Field(repr=False)
    empreintes_cles: frozenset[str] = Field(repr=False)

    @classmethod
    def depuis_environnement(cls) -> "Secrets":
        manquantes = [v for v in ("VLLM_URL", "VLLM_API_KEY", "CHSA_API_KEYS") if not os.environ.get(v)]
        if manquantes:
            raise RuntimeError(f"variables d'environnement manquantes : {', '.join(manquantes)}")
        empreintes = frozenset(e.strip().lower() for e in os.environ["CHSA_API_KEYS"].split(",") if e.strip())
        invalides = [e[:8] for e in empreintes if not SHA256.fullmatch(e)]
        if invalides:
            raise RuntimeError(f"CHSA_API_KEYS doit contenir des empreintes SHA-256 (64 caractères "
                               f"hexadécimaux) ; invalides : {invalides}")
        return cls(vllm_url=os.environ["VLLM_URL"].rstrip("/"),
                   vllm_api_key=os.environ["VLLM_API_KEY"],
                   empreintes_cles=empreintes)
