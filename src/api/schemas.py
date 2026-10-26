"""Contrat de l'API de la passerelle CHSA (schémas Pydantic v2).

Les valeurs qui dépendent du déploiement (modèles autorisés, longueur maximale de la question,
échelle de priorité) restent dans configs/serving.yaml : elles sont contrôlées par l'application,
pas figées ici. Ce module ne lit aucun fichier et n'a aucune dépendance au code d'entraînement.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RequeteQuestion(BaseModel):
    """Question envoyée par un utilisateur."""

    model_config = ConfigDict(extra="forbid")   # un champ inconnu est une erreur, pas un oubli silencieux

    question: str = Field(description="Question ou cas clinique, en texte libre.")
    modele: str | None = Field(
        default=None,
        description="Modèle servi (« chsa-sft » par défaut, « chsa-dpo » sur demande explicite).",
    )
    inclure_priorite: bool = Field(
        default=True,
        description="Demander aussi le niveau de priorité (valeur NON VALIDÉE, voir `Priorite`).",
    )

    @field_validator("question")
    @classmethod
    def question_non_vide(cls, valeur: str) -> str:
        valeur = valeur.strip()
        if not valeur:
            raise ValueError("la question est vide")
        return valeur


class Priorite(BaseModel):
    """Niveau de priorité proposé par le modèle — jamais validé cliniquement.

    Le modèle n'a pas appris le triage : cette valeur ne doit pas orienter une prise en charge.
    `format_valide` vaut False quand la sortie du modèle ne désigne aucune option reconnue ;
    `niveau` est alors None.
    """

    niveau: str | None = Field(description="Option choisie, par exemple « Tri 2 ».")
    echelle: str = Field(description="Échelle de référence et adaptation retenue.")
    statut: Literal["non_validee"] = "non_validee"
    format_valide: bool


class ReponseQuestion(BaseModel):
    """Réponse de la passerelle : texte du modèle et éléments de traçabilité."""

    id_interaction: str = Field(description="Identifiant de la ligne du journal.")
    modele: str = Field(description="Nom du modèle servi.")
    revision: str = Field(description="Révision des poids (SHA du Hugging Face Hub).")
    reponse: str
    complete: bool = Field(description="False si la génération a atteint la limite de tokens.")
    raison_fin: str = Field(description="finish_reason renvoyé par vLLM (« stop » ou « length »).")
    priorite: Priorite | None = Field(description="None si elle n'a pas été demandée.")
    avertissement: str
    latence_ms: int
    tokens_prompt: int
    tokens_reponse: int


class ModeleServi(BaseModel):
    nom_servi: str
    repo: str
    revision: str
    par_defaut: bool


class Sante(BaseModel):
    """État de la passerelle et du serveur vLLM."""

    statut: Literal["ok", "degrade"] = Field(description="« degrade » si vLLM ne répond pas.")
    vllm: Literal["ok", "indisponible"]
    modeles: list[ModeleServi]
