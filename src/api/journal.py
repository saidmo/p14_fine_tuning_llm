"""Journal des interactions de la passerelle (SQLite, minimisé).

Traçabilité : chaque requête, réussie ou en erreur, laisse une ligne reliée à la réponse
renvoyée par son identifiant (`id_interaction`).

Minimisation (RGPD) : la table n'a AUCUNE colonne pour l'adresse IP ni pour la clé ;
l'utilisateur n'est identifié que par les 8 premiers caractères de l'empreinte de sa clé.
Les lignes plus anciennes que la durée de conservation sont supprimées à chaque écriture.
Le POC ne reçoit que des questions de test : aucune donnée réelle de patient.

Les fonctions sont synchrones (sqlite3) ; l'application les appelle hors de la boucle
asynchrone (run_in_threadpool). Une connexion par opération : simple et sûr entre threads.
"""
import json
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id_interaction          TEXT PRIMARY KEY,
    horodatage              TEXT NOT NULL,     -- ISO 8601, UTC
    cle                     TEXT NOT NULL,     -- 8 premiers caractères de l'empreinte SHA-256
    modele                  TEXT NOT NULL,
    revision                TEXT NOT NULL,
    parametres              TEXT NOT NULL,     -- JSON : paramètres de génération envoyés à vLLM
    question                TEXT NOT NULL,
    statut                  TEXT NOT NULL CHECK (statut IN ('ok', 'erreur')),
    erreur                  TEXT,
    reponse                 TEXT,
    complete                INTEGER,
    raison_fin              TEXT,
    priorite_niveau         TEXT,
    priorite_format_valide  INTEGER,
    priorite_brute          TEXT,              -- sortie brute du modèle, pour auditer l'extraction
    latence_ms              INTEGER,
    tokens_prompt           INTEGER,
    tokens_reponse          INTEGER
);
CREATE INDEX IF NOT EXISTS idx_interactions_horodatage ON interactions (horodatage);
"""


@dataclass
class Interaction:
    id_interaction: str
    cle: str
    modele: str
    revision: str
    parametres: dict
    question: str
    statut: str = "ok"
    erreur: str | None = None
    reponse: str | None = None
    complete: bool | None = None
    raison_fin: str | None = None
    priorite_niveau: str | None = None
    priorite_format_valide: bool | None = None
    priorite_brute: str | None = None
    latence_ms: int | None = None
    tokens_prompt: int | None = None
    tokens_reponse: int | None = None
    horodatage: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Journal:
    def __init__(self, chemin: str | Path, conservation_jours: int):
        self.chemin = Path(chemin)
        self.conservation = timedelta(days=conservation_jours)
        self.chemin.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connexion()) as cnx, cnx:
            cnx.execute("PRAGMA journal_mode=WAL")   # lectures possibles pendant une écriture
            cnx.executescript(SCHEMA)

    def _connexion(self) -> sqlite3.Connection:
        return sqlite3.connect(self.chemin, timeout=10)

    def ecrire(self, interaction: Interaction) -> None:
        ligne = asdict(interaction)
        ligne["parametres"] = json.dumps(ligne["parametres"], ensure_ascii=False, sort_keys=True)
        colonnes = ", ".join(ligne)
        marqueurs = ", ".join(f":{c}" for c in ligne)
        limite = (datetime.now(timezone.utc) - self.conservation).isoformat()
        with closing(self._connexion()) as cnx, cnx:          # une transaction : purge + insertion
            cnx.execute("DELETE FROM interactions WHERE horodatage < ?", (limite,))
            cnx.execute(f"INSERT INTO interactions ({colonnes}) VALUES ({marqueurs})", ligne)

    def lire(self, id_interaction: str) -> dict | None:
        """Ligne du journal correspondant à une réponse (audit, tests)."""
        with closing(self._connexion()) as cnx:
            cnx.row_factory = sqlite3.Row
            ligne = cnx.execute("SELECT * FROM interactions WHERE id_interaction = ?",
                                (id_interaction,)).fetchone()
        return dict(ligne) if ligne else None
