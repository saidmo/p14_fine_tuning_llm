"""Anonymisation des jeux sélectionnés avec Presidio (paramètres : configs/anonymisation.yaml).

Principe : des règles précises MASQUENT, la NER spaCy (fort rappel, faible précision
mesurée) sert uniquement à l'AUDIT des noms restants.

Détection (masquage) :
  - civilité + nom : "Monsieur D. Daniel", "madame S. Fatima" (la civilité est conservée) ;
  - lexique de prénoms (fr) : "Cas clinique : Marianne G. 20 ans" ;
  - contextes (en/fr) : présentation ("My name is…", "Myself…"), salutation ("Hi Anita,"),
    signature ("Thank you, ali") ;
  - coordonnées : emails, téléphones, URL explicites.
Les éponymes médicaux ("maladie d'Addison", "Lasilix®") sont protégés par une règle de
contexte ; un prénom collé à un trait d'union n'est pas masqué (Charcot-Marie-Tooth).
Lieux et âges sont conservés (information clinique).

Paires ChatDoctor (vrais messages de patients), AVANT le masquage ci-dessus :
  réduction structurelle, généralisation des dates et masquage des lieux confirmés
  (src/data/reduce_chatdoctor.py). Vérification préalable fermée par défaut : s'il reste
  un candidat de lieu sans décision humaine, rien n'est écrit.

Autres mesures :
  - dedup_key (texte brut normalisé) remplacée par son empreinte sha256 ;
  - meta.golden_answer (UltraMedical) supprimée ;
  - l'échantillon de relecture (avec les noms d'origine) reste dans data/interim/,
    non versionné ; reports/ ne contient que des comptages.

Sorties :
    data/interim/anonymise/<jeu>.jsonl
    data/interim/anonymise/revue_<jeu>.md    (relecture manuelle, NON versionné)
    reports/anonymisation.json

Usage (depuis la racine du projet) :
    python -m src.data.anonymize --demo           # contrôle de la configuration
    python -m src.data.anonymize --only eval_clinique
    python -m src.data.anonymize
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import spacy
import yaml
from presidio_analyzer import (AnalyzerEngine, EntityRecognizer, Pattern, PatternRecognizer,
                               RecognizerRegistry, RecognizerResult)
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import EmailRecognizer, PhoneRecognizer, SpacyRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from src.data import reduce_chatdoctor as rcd
from src.data.sources import ROOT

CONFIG_PATH = ROOT / "configs" / "anonymisation.yaml"
PRIVATE_PATH = ROOT / "configs" / "prive" / "donnees_sensibles.yaml"   # non versionné
INPUT_DIR = ROOT / "data" / "interim" / "selection"
OUTPUT_DIR = ROOT / "data" / "interim" / "anonymise"
REPORT_PATH = ROOT / "reports" / "anonymisation.json"

DATASETS = ["sft_train", "sft_validation", "sft_test", "eval_clinique",
            "dpo_train", "dpo_validation", "dpo_test"]
TEXT_FIELDS = ("instruction", "response", "prompt", "chosen", "rejected")
SEED = 42
RULE_RECOGNIZERS = ("CivilityRecognizer", "LexiconRecognizer", "ContextRecognizer")

# URL explicites uniquement : le reconnaisseur Presidio accepte les domaines nus et prend
# des phrases collées ("dîner.La", "E.Coli") pour des domaines (.la, .co sont des TLD).
URL_EXPLICIT = r"\b(?:https?://|www\.)[^\s<>()\[\]\"']+(?<![.,;:!?])"   # sans ponctuation ni crochet final

DEMO = [
    ("fr", "Monsieur D. Daniel, âgé de 38 ans, forestier au Gabon (Lambaréné), "
           "présente une maladie d'Addison."),
    ("fr", "Mme B.A, 52 ans, test de Coombs direct négatif. Traitement par Lasilix®. "
           "Contact : jean.dupont@exemple.fr"),
    ("en", "Hi Dr. Smith, I am John, 45 years old, and I was diagnosed with Parkinson's disease. "
           "Call me at +1 650 253 0000."),
    ("fr", "Rappelez Madame Martin au 06 12 34 56 78."),
    ("fr", "Cas clinique : Yvan Y, 17 ans. Monsieur M., 45 ans, et Mme M. Selon son épouse, "
           "M. M. va bien. Mme aH. aussi."),
    ("fr", "Cas clinique : Marianne G. 20 ans. Marie et Jean ont un fils, Jean-Louis. "
           "Vous voyez madame S. Fatima. Maladie de Charcot-Marie-Tooth, maladie de Pierre Marie, "
           "réaction de Waaler-Rose, syndrome de Claude Bernard-Horner."),
    ("fr", "Prise en charge de Monsieur G.\n\nQuestion : que faites-vous chez Monsieur G. Quelles "
           "sont vos priorités ? M. Le V. Joseph est veuf."),
    ("fr", "Melle Pamella A, traitée par Augmentin. Choc septique à E.Coli après le dîner.La "
           "patiente a consulté www.exemple.fr. Manœuvres de Lasègue positives."),
    ("en", "Respected Sir, Myself Anita Verma from Pune. My name is ali and I have pain. "
           "Thank you, ali"),
    ("en", "Hi Anita, the answer is A. Measles B. German measles. Metformin helps. "
           "Hello Doctor, thanks doctor\nRegards,\nJohn Smith"),
    ("en", "Discuss it with your doctor. They can help. Dr. Thanks for your reply. "
           "Thanks in Advance. Regards, Ramesh (India)"),
    ("en", "hello sir i am rohit, my friend mohan said John Smith and Sarah had the same. "
           "See [www.kidney.org](http://www.kidney.org). Bell's palsy, Will power, May 2020."),
    ("fr", "Il attend les fêtes de Noël. Monsieur LUC...Jean, 64 ans. Enfant Z. Aicha, 9 ans."),
    ("en", "Hello Omar from Thane. I hope this helps, Nisha. Your hair loss, Vikas."),
]


# ---------------------------------------------------------------- motifs (fonctions pures)
_UPPER, _LOWER = "A-ZÀ-Ý", "a-zà-ÿ"
_INITIAL = rf"[{_UPPER}](?:\.|(?![{_LOWER}]))"            # "G." ou "Y" (pas "Daniel")
_NAME_PART = (rf"(?:[{_UPPER}]\.(?:\s?-?{_INITIAL})*"      # initiales : R. / B.A / J.-P.
              rf"|[{_UPPER}][{_LOWER}'’-]+"                # nom capitalisé : Daniel
              rf"|[{_UPPER}]{{2,}}\b"                      # nom en capitales : DUPONT
              rf"|[{_UPPER}](?![{_LOWER}]))")              # initiale seule : R
_WORD = rf"[A-Za-zÀ-ÿ][{_LOWER}'’-]+"                      # mot, capitalisé ou non
_CAP = rf"[{_UPPER}][{_LOWER}'’-]+"                        # mot capitalisé


def _stop_lookahead(words: list[str]) -> str:
    return rf"(?!(?:{'|'.join(re.escape(w) for w in words)})\b)" if words else ""


def civility_alternation(civilites: list[str], casse_libre: list[str] | None = None) -> str:
    """Alternative regex des civilités, les plus longues d'abord ("Mme" avant "M.").
    Seules les civilités de `casse_libre` sont insensibles à la casse ("madame")."""
    free = set(casse_libre or [])
    parts = []
    for c in sorted(civilites, key=len, reverse=True):
        token = rf"(?i:{re.escape(c)})" if c in free else re.escape(c)
        parts.append(token if c.endswith(".") else rf"{token}\.?")
    return "|".join(parts)


def civility_pattern(civilites: list[str], non_noms: list[str] | None = None,
                     casse_libre: list[str] | None = None) -> re.Pattern:
    """Civilité + 1 à 3 éléments de nom ; seul le groupe 'nom' est masqué.
    Séparateur : espace uniquement (jamais un saut de ligne) ; un mot de `non_noms`
    ne peut faire partie du nom, ni en 1re position ("Dr. Thanks") ni ensuite
    ("Monsieur G. Quelles sont…")."""
    stop = _stop_lookahead(non_noms or [])
    return re.compile(rf"\b(?:{civility_alternation(civilites, casse_libre)}) +"
                      rf"(?P<nom>{stop}{_NAME_PART}(?: {stop}{_NAME_PART}){{0,2}})")


def lexicon_pattern(names: list[str], non_noms: list[str], casse_libre: bool = False) -> re.Pattern:
    """Prénom du lexique (composés acceptés : Jean-Louis), suivi éventuellement d'un nom
    capitalisé et/ou d'une initiale. Jamais collé à un trait d'union ou à une lettre :
    "Charcot-Marie-Tooth", "Waaler-Rose" ne sont pas touchés."""
    alt = "|".join(re.escape(n) for n in sorted(set(names), key=len, reverse=True))
    if casse_libre:
        alt = rf"(?i:{alt})"
    first = rf"(?:{alt})(?:-(?:{alt}))*"
    surname = rf"(?: {_stop_lookahead(non_noms)}{_CAP})?"
    return re.compile(rf"(?<![\w-])(?P<nom>{first}{surname}(?: {_INITIAL})?)(?![\w-])")


CONTEXT_PATTERNS: dict[str, list[str]] = {
    "en": [
        rf"(?i:\bmy name is)\s+(?P<nom>{_WORD}(?: {_CAP})?)",                 # "my name is ali"
        rf"(?i:\b(?:myself|this is))\s+(?P<nom>{_CAP}(?: {_CAP})?)",          # "Myself Anita Verma"
        rf"(?i:\b(?:hi|hello|hey|dear))\s+(?P<nom>{_CAP})(?=\s*[,!:]|\s+from\b)",  # "Hi Anita," / "Hello Omar from"
        rf"(?i:\b(?:thanks|thank you|regards|best regards|warm regards|sincerely|cheers))"
        rf"[,.!]?\s+(?P<nom>{_WORD}(?: {_CAP})?)[ \t]*(?:\([^)\n]{{0,30}}\))?[ \t]*[.!]?[ \t]*$",  # "Regards, Ramesh (India)"
    ],
    "fr": [
        rf"(?i:\bje m['’]appelle)\s+(?P<nom>{_WORD}(?: {_CAP})?)",
        rf"(?i:\b(?:bonjour|chère|cher))\s+(?P<nom>{_CAP})(?=\s*[,!:])",
    ],
}


def eponym_patterns(cfg: dict[str, Any]) -> tuple[re.Pattern, re.Pattern]:
    before = "|".join(re.escape(w) for w in cfg["mots_avant"])
    after = "|".join(re.escape(w) for w in cfg["mots_apres"])
    return (re.compile(rf"(?:\b(?:{before})\s+(?:de\s+(?:la\s+)?|d['’]\s*|du\s+|des\s+|of\s+(?:the\s+)?)?"
                       # nom composé à tiret espacé : "Creutzfeld – Jacob", "Guillain - Barré"
                       rf"|\b[A-ZÀ-Ý][a-zà-ÿ]+\s*[–—-]\s*)$", re.IGNORECASE),
            # parenthèse courte tolérée : "the John Cunningham (JC) virus"
            re.compile(rf"^['’]?s?\s+(?:\([^)\n]{{1,15}}\)\s+)?(?:{after})\b", re.IGNORECASE))


def load_lexicon(cfg: dict[str, Any], lang: str) -> list[str]:
    path = cfg.get("lexique_prenoms", {}).get(lang)
    if not path:
        return []
    lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


# ---------------------------------------------------------------- reconnaisseur
class GroupRegexRecognizer(EntityRecognizer):
    """PERSON détecté par des regex dont seul le groupe 'nom' est l'entité : le reste du
    motif (civilité, formule de politesse) sert de contexte et n'est pas masqué."""

    def __init__(self, name: str, language: str, patterns: list[re.Pattern],
                 stopwords: list[str] | None = None) -> None:
        self.patterns = patterns
        self.stopwords = {w.lower() for w in stopwords or []}
        super().__init__(supported_entities=["PERSON"], supported_language=language, name=name)

    def load(self) -> None:
        pass

    def analyze(self, text: str, entities: list[str], nlp_artifacts=None) -> list[RecognizerResult]:
        if "PERSON" not in entities:
            return []
        metadata = {RecognizerResult.RECOGNIZER_NAME_KEY: self.name,
                    RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: self.id}
        results = []
        for pattern in self.patterns:
            for m in pattern.finditer(text):
                if m.group("nom").split()[0].strip(".,").lower() in self.stopwords:
                    continue
                results.append(RecognizerResult("PERSON", m.start("nom"), m.end("nom"), 0.85,
                                                recognition_metadata=metadata))
        return results


# ---------------------------------------------------------------- moteur
class Anonymizer:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        languages = list(cfg["modeles_spacy"])
        nlp_engine = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": lang, "model_name": model}
                       for lang, model in cfg["modeles_spacy"].items()]}).create_engine()

        # Registre explicite : on sait exactement quels reconnaisseurs tournent, par langue
        registry = RecognizerRegistry(supported_languages=languages)
        for lang in languages:
            non_noms = cfg["mots_non_noms"][lang]
            if lang in cfg["ner_spacy"]:
                registry.add_recognizer(SpacyRecognizer(supported_language=lang,
                                                        supported_entities=["PERSON"]))
            casse_libre = cfg["civilites_casse_libre"].get(lang, [])
            registry.add_recognizer(GroupRegexRecognizer(
                f"CivilityRecognizer_{lang}", lang,
                [civility_pattern(cfg["civilites"][lang], non_noms, casse_libre)]))
            # Un prénom n'a pas de langue : chaque texte consulte les deux lexiques — le sien
            # selon ses règles de casse, l'autre sensible à la casse ("Guy" ne doit pas masquer
            # "a guy" dans un texte anglais). Ex. : "My brother Roland", "Le jeune John T."
            lexicon = load_lexicon(cfg, lang)
            others = sorted({n for other in languages if other != lang
                             for n in load_lexicon(cfg, other)} - set(lexicon))
            if lexicon:
                registry.add_recognizer(GroupRegexRecognizer(
                    f"LexiconRecognizer_{lang}", lang,
                    [lexicon_pattern(lexicon, non_noms, lang in cfg["lexique_casse_libre"])]))
            if others:
                registry.add_recognizer(GroupRegexRecognizer(
                    f"LexiconRecognizer_autres_{lang}", lang, [lexicon_pattern(others, non_noms)]))
            if cfg["liste_noire"].get(lang):         # formes atypiques : casse exacte
                registry.add_recognizer(GroupRegexRecognizer(
                    f"LexiconRecognizer_listenoire_{lang}", lang,
                    [lexicon_pattern(cfg["liste_noire"][lang], non_noms)]))
            if CONTEXT_PATTERNS.get(lang):
                registry.add_recognizer(GroupRegexRecognizer(
                    f"ContextRecognizer_{lang}", lang,
                    [re.compile(p, re.MULTILINE) for p in CONTEXT_PATTERNS[lang]],
                    cfg["mots_non_noms_contexte"].get(lang, [])))
            registry.add_recognizer(EmailRecognizer(supported_language=lang))
            registry.add_recognizer(PatternRecognizer(
                supported_entity="URL", supported_language=lang, name=f"ExplicitUrlRecognizer_{lang}",
                patterns=[Pattern("url_explicite", URL_EXPLICIT, 0.6)]))
            registry.add_recognizer(PhoneRecognizer(supported_language=lang,
                                                    supported_regions=cfg["regions_telephone"]))

        self.analyzer = AnalyzerEngine(registry=registry, nlp_engine=nlp_engine,
                                       supported_languages=languages)
        self.engine = AnonymizerEngine()
        self.operators = {lang: {entity: OperatorConfig("replace", {"new_value": tag})
                                 for entity, tag in tags.items()}
                          for lang, tags in cfg["remplacements"].items()}
        self.before, self.after = eponym_patterns(cfg["eponymes"])
        self.whitelist = {w.lower() for w in cfg["eponymes"]["liste_blanche"]}
        # spaCy inclut souvent la civilité dans ses entités : "Monsieur D. Daniel"
        self.civility_prefix = {
            lang: re.compile(rf"^(?:{civility_alternation(civ, cfg['civilites_casse_libre'].get(lang))})(?:\s+|$)")
            for lang, civ in cfg["civilites"].items()}
        # Audit résiduel : NER spaCy seule, appliquée au texte déjà anonymisé
        self.audit_nlp = {lang: spacy.load(cfg["modeles_spacy"][lang])
                          for lang in cfg["audit_residuel"]["langues"]}
        # Réduction structurelle ChatDoctor : mêmes règles que l'outil de mesure reduce_chatdoctor
        self.cd_rules = rcd.build_rules(rcd.load_names(rcd.LEXICON_PATH))
        self.cd_places = rcd.load_places_config(rcd.PLACES_CONFIG_PATH)
        self.cd_nlp = self.audit_nlp.get("en") or spacy.load(rcd.SPACY_MODEL)

    def reduce_chatdoctor(self, record: dict[str, Any], decided: dict[str, list]) -> Counter:
        """Réduction, dates, lieux : appliqués aux champs texte d'une paire ChatDoctor."""
        counts: Counter = Counter()
        for field in TEXT_FIELDS:
            text = record.get(field)
            if not isinstance(text, str) or not text:
                continue
            reduced, spans = rcd.reduce_text(text, self.cd_rules)
            dated, dates = rcd.generalize_dates(reduced)
            final, places = rcd.mask_places(dated, decided["confirmes"])
            record[field] = final
            counts.update(r for rule, _, _ in spans for r in rule.split("+"))
            counts["date_generalisee"] += len(dates)
            counts["lieu_masque"] += len(places)
        return counts

    def trim_civility(self, text: str, r: RecognizerResult, lang: str) -> bool:
        """Retire la civilité en tête d'une entité PERSON issue de spaCy (conservée dans
        le texte). Retourne False si l'entité ne contenait que la civilité."""
        source = (r.recognition_metadata or {}).get(RecognizerResult.RECOGNIZER_NAME_KEY, "")
        if source.startswith(RULE_RECOGNIZERS):
            return True          # jamais de civilité dans ces entités ; "M." y est une initiale
        match = self.civility_prefix[lang].match(text[r.start:r.end])
        if match:
            r.start += match.end()
        return r.start < r.end

    def is_eponym(self, text: str, r: RecognizerResult) -> bool:
        entity, following = text[r.start:r.end], text[r.end:r.end + 40]
        return (entity.lower() in self.whitelist
                or entity.rstrip().endswith(("®", "™"))      # spaCy peut inclure le ®
                or following.startswith(("®", "™"))
                or bool(self.before.search(text[max(0, r.start - 40):r.start]))
                or bool(self.after.search(following)))

    def __call__(self, text: str, lang: str) -> tuple[str, list[RecognizerResult], list[RecognizerResult]]:
        thresholds = self.cfg["seuils"]
        results = self.analyzer.analyze(text=text, language=lang, entities=list(thresholds),
                                        score_threshold=min(thresholds.values()))
        results = [r for r in results if r.score >= thresholds[r.entity_type]]
        kept, neutralised = [], []
        for r in results:
            if r.entity_type == "PERSON":
                if not self.trim_civility(text, r, lang):
                    continue
                if self.is_eponym(text, r):
                    neutralised.append(r)
                    continue
            kept.append(r)
        if not kept:
            return text, kept, neutralised
        new_text = self.engine.anonymize(text=text, analyzer_results=kept,
                                         operators=self.operators[lang]).text
        return new_text, kept, neutralised


def snippet(text: str, r: RecognizerResult, label: str) -> str:
    extract = f"{text[max(0, r.start - 60):r.start]}⟦{text[r.start:r.end]}⟧{text[r.end:r.end + 60]}"
    return f"- `{r.entity_type}` … {extract.replace(chr(10), ' ')} … → {label}"


# ---------------------------------------------------------------- audit résiduel
def audit_residual(anonymizer: Anonymizer, pool: list[tuple[str, str]],
                   rng: random.Random) -> dict[str, Any]:
    """Applique la NER spaCy à un échantillon de textes anonymisés et liste les entités
    PER restantes (candidats à vérifier : surtout des faux positifs attendus)."""
    sample = rng.sample(pool, min(anonymizer.cfg["audit_residuel"]["echantillon"], len(pool)))
    extraits: list[str] = []
    seen: set[str] = set()
    for lang, text in sample:
        tags = set(anonymizer.cfg["remplacements"][lang].values()) | {rcd.PLACE_TAG}
        civilites = {c.rstrip(".").lower() for c in anonymizer.cfg["civilites"][lang]}
        for ent in anonymizer.audit_nlp[lang](text).ents:
            candidate = ent.text.strip()
            if (ent.label_ not in ("PER", "PERSON") or any(t in candidate for t in tags)
                    or candidate.rstrip(".").lower() in civilites):   # civilité seule : bruit
                continue
            context = text[max(0, ent.start_char - 50):ent.end_char + 50].replace("\n", " ")
            line = f"- … {context} … → candidat : ⟦{candidate}⟧"
            if line not in seen:                                     # cas cliniques partagés
                seen.add(line)
                extraits.append(line)
    return {"textes_audites": len(sample), "candidats": len(extraits), "extraits": extraits}


# ---------------------------------------------------------------- traitement
def chatdoctor_preflight(anonymizer: Anonymizer, names: list[str]) -> dict[str, dict[str, list]]:
    """Classe les lieux de tous les prompts ChatDoctor AVANT toute écriture.
    Fermé par défaut : un seul candidat sans décision humaine arrête le traitement."""
    decisions: dict[str, dict[str, list]] = {}
    pending: dict[str, str] = {}
    for name in names:
        with (INPUT_DIR / f"{name}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record.get("origine") != rcd.ORIGINE:
                    continue
                decided = rcd.classify_places(record.get("prompt") or "", anonymizer.cd_nlp,
                                              anonymizer.cd_places)
                decisions[record["id"]] = decided
                for candidate, context in decided["a_classer"]:
                    pending.setdefault(candidate, context.replace("\n", " "))
    if pending:
        listing = "\n".join(f"  - {c} : …{ctx}…" for c, ctx in pending.items())
        raise SystemExit(f"{len(pending)} candidat(s) de lieu sans décision (fermé par défaut) :\n"
                         f"{listing}\n→ les classer dans {rcd.PLACES_CONFIG_PATH.name}, puis relancer.")
    return decisions


def process_dataset(name: str, anonymizer: Anonymizer, sample_size: int,
                    decisions: dict[str, dict[str, list]]) -> dict[str, Any]:
    stats = {"enregistrements": 0, "enregistrements_modifies": 0,
             "entites_masquees": Counter(), "eponymes_neutralises": 0,
             "paires_chatdoctor": 0, "reduction_chatdoctor": Counter()}
    replaced_snippets, neutralised_snippets = [], []
    audit_pool: list[tuple[str, str]] = []           # (langue, texte anonymisé)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with (INPUT_DIR / f"{name}.jsonl").open(encoding="utf-8") as src, \
         (OUTPUT_DIR / f"{name}.jsonl").open("w", encoding="utf-8") as out:
        for line in src:
            record = json.loads(line)
            lang = record["langue"]
            counts: Counter = Counter()
            reduction: Counter = Counter()
            if record.get("origine") == rcd.ORIGINE:      # personnes réelles : avant le masquage
                reduction = anonymizer.reduce_chatdoctor(record, decisions[record["id"]])
                stats["paires_chatdoctor"] += 1
                stats["reduction_chatdoctor"].update(reduction)
            for field in TEXT_FIELDS:
                text = record.get(field)
                if not isinstance(text, str) or not text:
                    continue
                new_text, kept, neutralised = anonymizer(text, lang)
                record[field] = new_text
                if lang in anonymizer.audit_nlp:
                    audit_pool.append((lang, new_text))
                counts.update(r.entity_type for r in kept)
                stats["eponymes_neutralises"] += len(neutralised)
                replaced_snippets += [snippet(text, r, anonymizer.cfg["remplacements"][lang][r.entity_type])
                                      for r in kept]
                neutralised_snippets += [snippet(text, r, "conservé (éponyme)") for r in neutralised]

            # Le texte brut ne doit subsister nulle part : clé hachée, golden_answer supprimée
            record["dedup_key"] = hashlib.sha256(record["dedup_key"].encode("utf-8")).hexdigest()
            record.get("meta", {}).pop("golden_answer", None)
            record["anonymisation"] = dict(counts)
            if record.get("origine") == rcd.ORIGINE:
                record["reduction_chatdoctor"] = dict(reduction)
            record["transformations"] = record.get("transformations", []) + \
                (["reduction_chatdoctor"] if record.get("origine") == rcd.ORIGINE else []) + \
                (["anonymisation_presidio"] if counts else []) + ["cle_hachee"]

            stats["enregistrements"] += 1
            stats["enregistrements_modifies"] += bool(counts)
            stats["entites_masquees"].update(counts)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            if stats["enregistrements"] % 1000 == 0:
                print(f"    {stats['enregistrements']} enregistrements…")

    rng = random.Random(f"{SEED}:{name}")
    review = [f"# Relecture anonymisation — {name}", "",
              "Fichier de travail NON versionné : contient les valeurs d'origine.", "",
              f"## Masquages (échantillon de {min(sample_size, len(replaced_snippets))} "
              f"sur {len(replaced_snippets)})", ""]
    review += rng.sample(replaced_snippets, min(sample_size, len(replaced_snippets)))
    review += ["", f"## Éponymes conservés (échantillon de {min(sample_size, len(neutralised_snippets))} "
                   f"sur {len(neutralised_snippets)})", ""]
    review += rng.sample(neutralised_snippets, min(sample_size, len(neutralised_snippets)))

    residual = audit_residual(anonymizer, audit_pool, rng)
    review += ["", f"## Audit résiduel : entités PER détectées par spaCy après anonymisation "
                   f"({residual['textes_audites']} textes) — vérifier qu'aucune n'est un vrai nom", ""]
    review += residual["extraits"]
    (OUTPUT_DIR / f"revue_{name}.md").write_text("\n".join(review), encoding="utf-8")
    stats["audit_residuel"] = {k: v for k, v in residual.items() if k != "extraits"}

    return {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **stats, "entites_masquees": dict(stats["entites_masquees"]),
            "reduction_chatdoctor": dict(stats["reduction_chatdoctor"])}


def load_private() -> dict[str, Any]:
    """Paramètres sensibles (noms de personnes réelles) conservés hors du dépôt Git."""
    if not PRIVATE_PATH.exists():
        print(f"Avertissement : {PRIVATE_PATH} absent — liste noire et exclusions privées non appliquées.")
        return {}
    return yaml.safe_load(PRIVATE_PATH.read_text(encoding="utf-8")) or {}


def with_private(cfg: dict[str, Any]) -> dict[str, Any]:
    """Configuration d'exécution = configuration versionnée + liste noire privée.
    Le rapport ne recopie que la configuration versionnée."""
    private = load_private().get("liste_noire") or {}
    merged = {**cfg, "liste_noire": {lang: list(cfg["liste_noire"].get(lang) or []) + list(private.get(lang) or [])
                                     for lang in set(cfg["liste_noire"]) | set(private)}}
    return merged


def run_demo(anonymizer: Anonymizer) -> None:
    for lang, text in DEMO:
        new_text, kept, neutralised = anonymizer(text, lang)
        print(f"\n[{lang}] {text}\n  → {new_text}")
        for r in neutralised:
            print(f"  conservé (éponyme) : {text[r.start:r.end]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--demo", action="store_true", help="tester la configuration sur des phrases types")
    parser.add_argument("--only", choices=DATASETS, help="anonymiser un seul jeu")
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    anonymizer = Anonymizer(with_private(cfg))
    if args.demo:
        run_demo(anonymizer)
        return

    names = [args.only] if args.only else DATASETS
    print("[chatdoctor] vérification préalable des lieux…")
    decisions = chatdoctor_preflight(anonymizer, names)
    print(f"  {len(decisions)} paires ChatDoctor, aucun candidat de lieu sans décision")

    report = (json.loads(REPORT_PATH.read_text(encoding="utf-8"))
              if REPORT_PATH.exists() else {"jeux": {}})
    report["configuration"] = cfg
    for name in names:
        print(f"[{name}] anonymisation…")
        report["jeux"][name] = process_dataset(name, anonymizer, cfg["taille_echantillon_revue"], decisions)
        print(f"  modifiés : {report['jeux'][name]['enregistrements_modifies']} / "
              f"{report['jeux'][name]['enregistrements']} — {report['jeux'][name]['entites_masquees']}")

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Rapport : {REPORT_PATH}")


if __name__ == "__main__":
    main()
