"""Réduction structurelle des messages ChatDoctor (UltraMedical, origine "ChatDoctor").

ChatDoctor contient de vrais messages de patients : les données personnelles s'y
concentrent dans des zones sans valeur clinique (salutations, présentations,
provenance, signatures). Plutôt que de chercher les noms, on supprime ces zones ;
le corps du message (symptômes, langage familier) est conservé. Pour ces personnes
réelles, les quasi-identifiants (lieux, dates exactes) sont aussi traités (critère
de corrélation du G29).

Étapes, dans l'ordre :
  1. Suppression (règles prudentes : en cas de doute, on ne supprime pas)
     - salutation_ouverture  : "Hi doctor,", "Hello Nisha,", "Good evening, Omar."
     - salutation_nominative : "Hi John," après une fin de phrase
     - presentation          : "my name is rohit,", "Myself Anita Verma from Pune."
     - provenance            : "I am from Delhi"
     - signature             : "Thanks, Mary", "Jane Doe YYYY@YYYY", dernière ligne = prénom
  2. Généralisation des dates : le jour est retiré, mois et année conservés
     ("18.04.2012" -> "04.2012", "6th Feb 2014" -> "Feb 2014") ; date numérique
     ambiguë (jour et mois <= 12) -> année seule
  3. Masquage des lieux -> "[PLACE]" (paramètres : configs/lieux_chatdoctor.yaml)
     - masqués : les lieux CONFIRMÉS à la relecture, cherchés directement dans le prompt
     - la NER spaCy (GPE, LOC, FAC) et le motif d'établissement ne font que PROPOSER des
       candidats : pays, faux positifs, sigles et fragments écartés ; le reste est « à classer »
       (jamais masqué en silence : décision humaine requise)
     - dans les réponses : seuls les échos des lieux masqués dans le prompt

Cette étape MESURE seulement : les jeux ne sont pas modifiés.

Sorties :
    data/interim/reduction/revue_reduction_chatdoctor.md   (relecture, NON versionné)
    reports/reduction_chatdoctor.json                      (comptages uniquement)

Usage (depuis la racine du projet) :
    python -m src.data.reduce_chatdoctor
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import spacy
import yaml

from src.data.sources import ROOT

INPUT_DIR = ROOT / "data" / "interim" / "selection"
REVIEW_PATH = ROOT / "data" / "interim" / "reduction" / "revue_reduction_chatdoctor.md"
REPORT_PATH = ROOT / "reports" / "reduction_chatdoctor.json"
LEXICON_PATH = ROOT / "configs" / "prenoms_en.txt"
PLACES_CONFIG_PATH = ROOT / "configs" / "lieux_chatdoctor.yaml"
PRIVATE_PATH = ROOT / "configs" / "prive" / "donnees_sensibles.yaml"   # non versionné
SPACY_MODEL = "en_core_web_md"

DATASETS = ["dpo_train", "dpo_validation", "dpo_test"]
FIELDS = ("prompt", "chosen", "rejected")
ORIGINE = "ChatDoctor"
SEED = 42
REVIEW_SIZE = 30
DATE_SAMPLE = 40
MIN_PROMPT_CHARS = 40       # alerte : prompt presque vide après réduction
PLACE_TAG = "[PLACE]"
PLACE_LABELS = {"GPE", "LOC", "FAC"}

# Mots qui ne peuvent pas être un nom de famille après un prénom ("i am rohit and…")
_NOT_SURNAME = ("and|i|my|me|from|have|had|has|am|is|was|with|the|a|an|aged|age|years|yrs|"
                "year|old|suffering|having|want|need|plz|please|sir|doctor|dr|here|so|but|"
                "also|now|today|frm|wanted|would|will|can|could|got|get|this|that|in|on|at")
# Mots capitalisés qui ne sont pas un nom d'établissement ("My hospital", "The clinic")
_NOT_PLACE = ("My|The|This|That|A|An|Our|His|Her|Their|Your|Same|Local|Private|Government|Any|Which|"
              "Some|Many|Most|Other|Several|Large|Big|Good|Best|Top|Reputed|Research|Teaching|"
              "Avoiding|Shorter|Every|Each|Nearby|Nearest|Specialized|Specialised")


# ---------------------------------------------------------------- 1. suppression
def load_names(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def build_rules(names: list[str]) -> list[tuple[str, re.Pattern]]:
    lexicon = rf"(?i:{'|'.join(re.escape(n) for n in sorted(set(names), key=len, reverse=True))})"
    cap = r"[A-Z][a-z'’-]+"                                   # mot capitalisé
    word = r"[A-Za-z][A-Za-z'’-]+"                            # mot quelconque (2 lettres et +)
    name = rf"(?:{lexicon}|{cap})"
    surname = rf"(?:\s+(?!(?i:{_NOT_SURNAME})\b){word})?"     # "anita verma", pas "rohit and"
    title = r"(?:(?i:mr|mrs|ms|miss|dr|smt|shri|m)\.?\s*)"
    place = r"(?:\s+(?i:from)\s+[A-Za-z][A-Za-z .'-]{1,30}?(?=[,.!?\n]|\s+(?i:and)\b|$))"
    sep = r"[\s,.!:;-]*"
    greet = r"(?i:hi+|hello|hlw|hlo|helo|hey|dear|respected|namaste|good\s+(?:morning|afternoon|evening|day)|gud\s+\w+)"
    addressee = (rf"(?:\s+(?:(?i:dr\.?|doctor|doc|sir|madam|mam|ma'?am|all|there|team|everyone)"
                 rf"|{name}(?: {cap})?)"
                 rf"|,\s*{cap}(?: {cap})?(?=\s*[.,!:]))")          # "Good evening, Omar." (pas "Hi, My daughter")
    closing = (r"(?i:thanks|thank\s+you|thanking\s+you|many\s+thanks|regards|best\s+regards|"
               r"with\s+regards|warm\s+regards|kind\s+regards|sincerely|sincerelly|cheers|"
               r"yours\s+(?:truly|sincerely))(?:\s+(?i:very\s+much|a\s+lot|so\s+much|doctor|dr\.?|sir))?")
    # éléments admis dans une signature : nom, mot capitalisé, (pays), téléphone, email, "#"
    sig_token = (rf"(?:{lexicon}\b|[A-Z][\w'’.-]*|\([A-Za-z ]+\)|\d[\d -]{{5,}}\d|"
                 r"[\w.+-]+@[\w.-]+|#)")
    iam = r"(?i:\b(?:i\s*a\s*m|i'?m|i\s+m|this\s+is|it'?s|its))"         # "i am", "iam", "ia m", "IM"

    return [
        ("salutation_ouverture", re.compile(rf"^\s*(?:{greet}{addressee}?{place}?{sep})+")),
        ("salutation_nominative", re.compile(rf"(?<=[.!?\n])\s*{greet}\s+{name}(?: {cap})?{place}?\s*[,!:]\s*")),
        ("presentation", re.compile(rf"(?i:\bmy\s+name(?:\s+is)?)\s+{title}?(?!(?i:not|was|has)\b){word}{surname}{place}?{sep}")),
        ("presentation", re.compile(rf"(?i:\bmy\s*self)\s+{title}?{name}{surname}{place}?{sep}")),   # "My self X"
        ("presentation", re.compile(rf"{iam}\s+{title}?{lexicon}\b{surname}{place}?{sep}")),
        ("provenance", re.compile(rf"(?i:\b(?:i\s*am|i'?m|i\s+m|iam|we\s+are|we're))\s+(?i:from)\s+"
                                  rf"[A-Za-z][A-Za-z .'-]{{1,30}}?(?=[,.!?\n]|\s+(?i:and)\b|$){sep}")),
        ("signature", re.compile(rf"(?:(?<=[\s.!?,])|^){closing}{sep}(?:{sig_token}{sep}){{1,10}}\s*\Z")),
        # signature sans formule : nom + contact ("Jane Doe YYYY@YYYY") ou dernière ligne = prénom
        ("signature", re.compile(rf"(?<=[.!?\s])(?:{name}\s+){{1,3}}"
                                 rf"(?:YYYY@YYYY|[\w.+-]+@[\w.-]+|\+?\d[\d -]{{6,}}\d)\s*\Z")),
        ("signature", re.compile(rf"\n\s*{lexicon}\b(?:\s+{cap})?\s*[.!]?\s*\Z")),
    ]


def reduce_text(text: str, rules: list[tuple[str, re.Pattern]]) -> tuple[str, list[tuple[str, int, int]]]:
    """Retourne le texte réduit et les segments supprimés (règle(s), début, fin)."""
    spans = sorted(((name, m.start(), m.end()) for name, pattern in rules
                    for m in pattern.finditer(text) if m.end() > m.start()), key=lambda s: s[1])
    merged: list[tuple[str, int, int]] = []
    for name, start, end in spans:          # segments qui se chevauchent : fusionnés,
        if merged and start <= merged[-1][2]:  # avec la liste de toutes les règles en cause
            prev = merged[-1]
            rules_in = "+".join(sorted(set(prev[0].split("+")) | {name}))
            merged[-1] = (rules_in, prev[1], max(end, prev[2]))
        else:
            merged.append((name, start, end))

    out, cursor = [], 0
    for _, start, end in merged:
        out.append(text[cursor:start])
        cursor = end
    out.append(text[cursor:])
    reduced = re.sub(r"[ \t]{2,}", " ", "".join(out))
    reduced = re.sub(r"^[\s,.!:;-]+", "", reduced).strip()
    return reduced, merged


# ---------------------------------------------------------------- 2. dates
_MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
          r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?")   # capitalisés : "may" ≠ mai
_ORD = r"(?:st|nd|rd|th)?"


def _numeric_date(m: re.Match) -> str:
    """jour/mois/année ou mois/jour/année : on garde mois et année ; ambigu -> année seule."""
    a, sep, b, year = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
    if (a > 12 and b > 12) or a > 31 or b > 31 or a == 0 or b == 0:
        return m.group(0)                       # pas une date ("8-86-50")
    if a > 12:
        return f"{m.group(3)}{sep}{year}"       # 18.04.2012 -> 04.2012
    if b > 12:
        return f"{m.group(1)}{sep}{year}"       # 04/18/2012 -> 04/2012 (format US)
    return year                                 # 04/05/2012 : jour et mois indiscernables


def _month_year(m: re.Match) -> str:
    """Mois (+ année si présente), sans la virgule : "Feb 6, 2014" -> "Feb 2014"."""
    return m.group(1) + (f" {m.group(2).lstrip(', ')}" if m.group(2) else "")


DATE_RULES: list[tuple[str, re.Pattern, Callable[[re.Match], str]]] = [
    # (?<![\d.,]) : "1.3 April" est une valeur suivie d'un mois, pas le 3 avril
    ("date_jour_mois", re.compile(rf"(?<![\d.,])\b\d{{1,2}}{_ORD}\s+(?:of\s+)?({_MONTH})(,?\s+\d{{4}})?\b"),
     _month_year),                                                   # "6th Feb 2014" -> "Feb 2014"
    ("date_mois_jour", re.compile(rf"\b({_MONTH})\s+\d{{1,2}}{_ORD}\b(,?\s+\d{{4}})?"),
     _month_year),                                                   # "Feb 6, 2014" -> "Feb 2014"
    ("date_numerique", re.compile(r"\b(\d{1,2})([./-])(\d{1,2})\2(\d{4}|\d{2})\b"), _numeric_date),
]


def generalize_dates(text: str) -> tuple[str, list[tuple[str, str, str]]]:
    changes: list[tuple[str, str, str]] = []
    for rule, pattern, repl in DATE_RULES:
        def sub(m: re.Match, rule: str = rule, repl: Callable = repl) -> str:
            new = repl(m)
            if new != m.group(0):
                changes.append((rule, m.group(0), new))
            return new
        text = pattern.sub(sub, text)
    return text, changes


# ---------------------------------------------------------------- 3. lieux
# Mot de nom d'établissement : pas de point final, sauf abréviation courte ("St. Jude"),
# sinon le motif traverse une fin de phrase ("Bangalore. Some hospitals")
_FACILITY_WORD = r"(?:[A-Z][A-Za-z'&-]+|[A-Z][a-z]{0,2}\.)"
_FACILITY = re.compile(rf"\b(?!(?:{_NOT_PLACE})\b)((?:{_FACILITY_WORD}\s+){{1,4}})"
                       r"(?=(?i:hospitals?|clinic|nursing\s+home|medical\s+(?:college|cent(?:er|re))|"
                       r"institute|infirmary|school|college|university)\b|High\b)")


def _bounded(terms: list[str]) -> re.Pattern:
    """Recherche insensible à la casse, jamais à l'intérieur d'un mot. Un trait d'union
    ne bloque pas : "Manhattan-lower east side" doit être masqué (contrairement aux
    prénoms, il n'y a pas d'éponyme composé à protéger pour les lieux)."""
    alt = "|".join(re.escape(t) for t in sorted(set(terms), key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alt})(?!\w)", re.IGNORECASE)


def load_places_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    # Noms de personnes pris pour des lieux : conservés hors du dépôt Git
    private = (yaml.safe_load(PRIVATE_PATH.read_text(encoding="utf-8")) or {}) if PRIVATE_PATH.exists() else {}
    cfg["faux_positifs"] = list(cfg["faux_positifs"]) + list(private.get("lieux_faux_positifs") or [])
    confirmed = [str(p) for p in cfg["lieux_confirmes"]]
    return {"confirmes": _bounded(confirmed),
            "pays": {str(p).lower() for p in cfg["pays_conserves"]},
            "faux_positifs": {str(f).lower() for f in cfg["faux_positifs"]}}


def place_decision(candidate: str, cfg: dict[str, Any]) -> str | None:
    """Décision pour un candidat non confirmé : motif de non-masquage, ou None = à classer."""
    text = candidate.strip()
    low = text.lower()
    if low in cfg["pays"]:
        return "pays_ou_region"
    if low in cfg["faux_positifs"]:
        return "faux_positif"
    if re.fullmatch(r"[A-Z.]{1,5}", text):
        return "sigle"                                  # MVP, ENT, ST, LA (oreillette)…
    if len(text) < 3 or re.search(r"[\d*:+/\[?!(),;]|\.\w", text):
        return "forme"                                  # "baby.i", "drinks?if", "Duphaston 10"
    return None


def classify_places(original: str, nlp, cfg: dict[str, Any]) -> dict[str, list]:
    """Prompt D'ORIGINE (avant réduction : "my name rohit from Thane" compte aussi) :
    lieux confirmés présents, et décision pour chaque candidat de spaCy / du motif."""
    confirmed = [(m.start(), m.end()) for m in cfg["confirmes"].finditer(original)]
    candidates = [(m.start(1), m.end(1) - 1) for m in _FACILITY.finditer(original)]
    candidates += [(e.start_char, e.end_char) for e in nlp(original).ents if e.label_ in PLACE_LABELS]

    result: dict[str, list] = {"confirmes": [original[s:e] for s, e in confirmed],
                               "ecartes": [], "a_classer": []}
    for start, end in candidates:
        if any(s < end and start < e for s, e in confirmed):
            continue                                    # recouvre un lieu confirmé
        reason = place_decision(original[start:end], cfg)
        if reason:
            result["ecartes"].append((reason, original[start:end]))
        else:                                           # jamais masqué en silence : décision humaine
            result["a_classer"].append((original[start:end], original[max(0, start - 50):end + 50]))
    return result


def mask_places(text: str, places: list[str]) -> tuple[str, list[str]]:
    """Masque, dans un texte déjà réduit, les lieux confirmés trouvés dans le prompt d'origine.
    Sert au prompt comme aux réponses (échos)."""
    if not places:
        return text, []
    found: list[str] = []

    def sub(m: re.Match) -> str:
        found.append(m.group(0))
        return PLACE_TAG
    return _bounded(places).sub(sub, text), found


# ---------------------------------------------------------------- mesure
def highlight(text: str, spans: list[tuple[str, int, int]]) -> str:
    """Texte d'origine avec les segments supprimés encadrés ⟦− … ⟧."""
    parts, cursor = [], 0
    for _, start, end in spans:
        parts += [text[cursor:start], f"⟦−{text[start:end]}⟧"]
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts).replace("\n", " ⏎ ")


def main() -> None:
    names = load_names(LEXICON_PATH)
    rules = build_rules(names)
    lexicon_any = re.compile(rf"\b(?i:{'|'.join(re.escape(n) for n in sorted(set(names), key=len, reverse=True))})\b")
    places_cfg = load_places_config(PLACES_CONFIG_PATH)
    print(f"Chargement de {SPACY_MODEL}…")
    nlp = spacy.load(SPACY_MODEL)

    report: dict[str, Any] = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              "jeux": {}}
    examples: list[dict[str, Any]] = []
    all_places: Counter = Counter()
    all_excluded: Counter = Counter()
    to_classify: dict[str, list[str]] = {}
    all_dates: list[tuple[str, str, str]] = []

    for name in DATASETS:
        stats = {"paires_chatdoctor": 0, "paires_modifiees": 0, "regles": Counter(),
                 "caracteres_avant": 0, "caracteres_supprimes": 0, "prompts_trop_courts": 0,
                 "dates_generalisees": Counter(), "lieux_masques": 0, "lieux_echos": 0,
                 "lieux_ecartes": Counter(), "lieux_a_classer": 0,
                 "textes_avec_prenom_lexique_avant": 0, "textes_avec_prenom_lexique_apres": 0}
        with (INPUT_DIR / f"{name}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record.get("origine") != ORIGINE:
                    continue
                stats["paires_chatdoctor"] += 1
                # Lieux décidés sur le prompt D'ORIGINE : ceux qui disparaissent avec la réduction
                # ("from pune") doivent quand même être masqués s'ils reviennent dans une réponse
                decided = classify_places(record.get("prompt") or "", nlp, places_cfg)
                stats["lieux_ecartes"].update(reason for reason, _ in decided["ecartes"])
                stats["lieux_a_classer"] += len(decided["a_classer"])
                all_excluded.update(decided["ecartes"])
                for candidate, context in decided["a_classer"]:
                    to_classify.setdefault(candidate, []).append(context)

                modified, detail = False, {}
                for field in FIELDS:
                    text = record.get(field) or ""
                    reduced, spans = reduce_text(text, rules)
                    dated, date_changes = generalize_dates(reduced)
                    final, places = mask_places(dated, decided["confirmes"])
                    if field == "prompt":
                        stats["lieux_masques"] += len(places)
                        all_places.update(p.lower() for p in places)
                    else:
                        stats["lieux_echos"] += len(places)

                    stats["caracteres_avant"] += len(text)
                    stats["caracteres_supprimes"] += len(text) - len(reduced)
                    stats["regles"].update(r for rule, _, _ in spans for r in rule.split("+"))
                    stats["dates_generalisees"].update(rule for rule, _, _ in date_changes)
                    stats["textes_avec_prenom_lexique_avant"] += bool(lexicon_any.search(text))
                    stats["textes_avec_prenom_lexique_apres"] += bool(lexicon_any.search(final))
                    all_dates += date_changes
                    modified |= bool(spans or date_changes or places)
                    detail[field] = (text, final, spans, date_changes, places)
                stats["paires_modifiees"] += modified
                stats["prompts_trop_courts"] += len(detail["prompt"][1]) < MIN_PROMPT_CHARS
                examples.append({"jeu": name, "id": record["id"], **detail})

        for key in ("regles", "dates_generalisees", "lieux_ecartes"):
            stats[key] = dict(stats[key])
        stats["part_supprimee"] = round(stats["caracteres_supprimes"] / max(stats["caracteres_avant"], 1), 4)
        report["jeux"][name] = stats
        print(f"[{name}] {stats['paires_modifiees']} / {stats['paires_chatdoctor']} paires modifiées — "
              f"{stats['part_supprimee']:.1%} supprimés — dates : {stats['dates_generalisees']} — "
              f"lieux masqués : {stats['lieux_masques']} (+{stats['lieux_echos']} échos) — "
              f"à classer : {stats['lieux_a_classer']}")
    report["lieux_masques_uniques"] = len(all_places)
    report["candidats_ecartes_uniques"] = len(all_excluded)
    report["candidats_a_classer_uniques"] = len(to_classify)
    print(f"Candidats à classer (uniques) : {len(to_classify)}")

    rng = random.Random(SEED)
    review = ["# Relecture — réduction structurelle ChatDoctor", "",
              "Fichier de travail NON versionné : contient les textes d'origine.",
              "Segments supprimés encadrés ⟦− … ⟧ ; dates et lieux traités visibles dans le « après ».", "",
              "# 1. Paires tirées au hasard", ""]
    for ex in rng.sample(examples, min(REVIEW_SIZE, len(examples))):
        review += [f"## {ex['jeu']} — `{ex['id']}`", ""]
        prompt, final, spans, dates, places = ex["prompt"]
        review += ["**Prompt (avant)** : " + highlight(prompt, spans), "",
                   "**Prompt (après)** : " + final.replace("\n", " ⏎ "), ""]
        for field in ("chosen", "rejected"):
            text, _, spans, dates, places = ex[field]
            for rule, start, end in spans:
                context = (f"…{text[max(0, start - 40):start]}⟦−{text[start:end]}⟧"
                           f"{text[end:end + 40]}…").replace("\n", " ⏎ ")
                review.append(f"- *{field}* ({rule}) : {context}")
            review += [f"- *{field}* (date) : {old} → {new}" for _, old, new in dates]
            review += [f"- *{field}* (lieu, écho) : {p}" for p in places]
        review.append("")

    review += [f"# 2. À CLASSER — {len(to_classify)} candidats sans décision", "",
               "Pour chacun : ajouter à lieux_confirmes (c'est un lieu) ou à faux_positifs (ce n'en est pas un)",
               "dans configs/lieux_chatdoctor.yaml. À l'intégration, le pipeline s'arrête tant qu'il en reste.", ""]
    for candidate, contexts in sorted(to_classify.items(), key=lambda kv: -len(kv[1])):
        review.append(f"- **{candidate}** (×{len(contexts)}) : …{contexts[0].replace(chr(10), ' ')}…")
    review += ["", "# 2 bis. Lieux masqués (prompts) — liste exhaustive", ""]
    review += [f"- {place} (×{count})" for place, count in all_places.most_common()]
    review += ["", "# 2 ter. Candidats écartés — liste exhaustive (vérifier qu'aucun n'est un vrai lieu)", ""]
    review += [f"- {place_text} ({reason}, ×{count})"
               for (reason, place_text), count in all_excluded.most_common()]

    review += ["", f"# 3. Dates généralisées (échantillon de {min(DATE_SAMPLE, len(all_dates))} "
                   f"sur {len(all_dates)})", ""]
    review += [f"- ({rule}) {old} → {new}" for rule, old, new in rng.sample(all_dates, min(DATE_SAMPLE, len(all_dates)))]

    REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_PATH.write_text("\n".join(review), encoding="utf-8")
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Relecture : {REVIEW_PATH}\nRapport : {REPORT_PATH}")


if __name__ == "__main__":
    main()
