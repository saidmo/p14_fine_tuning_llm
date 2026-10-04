"""Schéma commun des exemples et fonctions de nettoyage de texte.

Deux types d'enregistrements :
  - SFTRecord : paire instruction -> réponse (QCM, QA ouverte, triage)
  - DPORecord : prompt + réponse préférée (chosen) + réponse rejetée (rejected)

Le formatage conversationnel (message system, chat template) n'est PAS fait ici :
c'est un choix d'entraînement, appliqué à l'assemblage, pour pouvoir le modifier
sans retraiter les données.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any

TASKS = {"mcq_single", "mcq_multi", "open_qa", "triage"}
LANGUES = {"fr", "en"}
SPLITS = {"train", "validation", "test", "unsplit"}   # unsplit : pas de split officiel
NIVEAUX_CONFIANCE = {"eleve", "moyen", "faible"}

# Libellés hétérogènes observés au profilage (MediQAl, config oeq)
QUESTION_TYPE_MAP = {"Raisonnement": "Reasoning"}
SUBJECT_MAP = {"Hématologie": "Hematology"}


# ---------------------------------------------------------------- nettoyage
_SPACES = re.compile(r"[ \t\u00a0\u202f]+")          # inclut les espaces insécables
_SPACE_BEFORE_NEWLINE = re.compile(r" +\n")
_MULTI_NEWLINES = re.compile(r"\n{3,}")
_REPEATED_PUNCT = re.compile(r"([?!])(?:\s*\1)+")   # "? ?" -> "?" ; les "..." sont préservés
_CORRUPTED_BULLET = re.compile(r"(?m)^\? +")      # "? texte" en début de ligne -> "- texte"

_CLEANING_STEPS = [
    ("unicode_nfc", lambda t: unicodedata.normalize("NFC", t)),
    ("fins_de_ligne", lambda t: t.replace("\r\n", "\n").replace("\r", "\n")),
    ("espaces_multiples", lambda t: _SPACES.sub(" ", t)),
    ("puces_corrompues", lambda t: _CORRUPTED_BULLET.sub("- ", t)),
    ("espaces_fin_de_ligne", lambda t: _SPACE_BEFORE_NEWLINE.sub("\n", t)),
    ("sauts_de_ligne_multiples", lambda t: _MULTI_NEWLINES.sub("\n\n", t)),
    ("ponctuation_repetee", lambda t: _REPEATED_PUNCT.sub(r"\1", t)),
    ("espaces_bordure", str.strip),
]




def clean_text(text: str | None, log: list[str]) -> str:
    """Nettoie un texte et ajoute à `log` le nom de chaque règle qui l'a modifié."""
    if text is None:
        return ""
    for name, rule in _CLEANING_STEPS:
        new = rule(text)
        if new != text:
            if name not in log:
                log.append(name)
            text = new
    return text


def normalize_label(value: str | None, mapping: dict[str, str], log: list[str]) -> str | None:
    """Harmonise un libellé de métadonnée (ex. 'Raisonnement' -> 'Reasoning')."""
    if value is None:
        return None
    value = value.strip()
    if value in mapping:
        if "libelle_normalise" not in log:
            log.append("libelle_normalise")
        return mapping[value]
    return value


def dedup_key(*parts: str | None) -> str:
    """Clé de comparaison : minuscules, sans accents ni ponctuation, espaces réduits.
    Même normalisation que le profilage, pour que les volumes restent comparables."""
    text = unicodedata.normalize("NFKD", " ".join(p or "" for p in parts))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------- enregistrements
def _check(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass
class _BaseRecord:
    id: str
    source: str
    config: str
    source_id: str
    source_revision: str
    langue: str
    split: str
    niveau_confiance: str
    dedup_key: str

    def _validate_base(self) -> None:
        _check(self.langue in LANGUES, "langue invalide")
        _check(self.split in SPLITS, "split invalide")
        _check(self.niveau_confiance in NIVEAUX_CONFIANCE, "niveau de confiance invalide")
        _check(self.dedup_key, "clé de dédoublonnage vide")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SFTRecord(_BaseRecord):
    task: str = ""
    instruction: str = ""
    response: str = ""
    medical_subject: str | None = None
    question_type: str | None = None
    has_clinical_case: bool = False
    # Métadonnées cliniques structurées : renseignées uniquement pour le bloc triage
    symptomes: list[str] | None = None
    antecedents: list[str] | None = None
    constantes: dict[str, Any] | None = None
    transformations: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate_base()
        _check(self.task in TASKS, "tâche invalide")
        _check(self.instruction, "instruction vide")
        _check(self.response, "réponse vide")


@dataclass
class DPORecord(_BaseRecord):
    prompt: str = ""
    chosen: str = ""
    rejected: str = ""
    label_type: str | None = None
    origine: str | None = None          # sous-source d'UltraMedical (préfixe de prompt_id)
    score_chosen: float | None = None
    score_rejected: float | None = None
    transformations: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate_base()
        _check(self.prompt, "prompt vide")
        _check(self.chosen and self.rejected, "réponse chosen ou rejected vide")
        _check(self.chosen != self.rejected, "chosen identique à rejected")
