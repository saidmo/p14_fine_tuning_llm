"""Contrôle exhaustif des noms après anonymisation (100 % des textes, 7 jeux).

La NER spaCy (fort rappel) est appliquée à TOUS les textes anonymisés. Ses entités
PER restantes passent des filtres automatiques documentés, puis chaque candidat
restant attend une décision humaine (configs/decisions_audit_noms.yaml) :
nom -> lexique ; pas un nom -> non_noms ; doute -> l'exemple est retiré.

Filtres automatiques (motif compté dans le rapport) :
  balise, civilite, forme, sigle_ou_code, marque (®/™), eponyme (règle de
  l'anonymisation), auteur_cite ("Kelley et al."), decision_non_nom, decision_exclusion,
  vocabulaire_minuscule : tous les mots du candidat apparaissent aussi en minuscules
  ailleurs dans le corpus ("Metformin"/"metformin") — sauf s'ils figurent dans un
  lexique de prénoms ("rohit" reste candidat) ; morphologie_medicale : mot isolé à
  suffixe médical (-itis, -emia, -mab…), hors lexique de prénoms.

Périmètres (decisions_audit_noms.yaml : revue_exhaustive) :
  - personnes_reelles : paires ChatDoctor (vrais messages de patients)
  - cas_fictifs_et_editoriaux : cas d'examen, QCM, pages NIH, contenus synthétiques
  "tout" (défaut, strict) : tout candidat attend une décision humaine ;
  "personnes_reelles" : seuls les candidats présents dans les textes ChatDoctor bloquent ;
  dans les deux cas, un prénom du lexique non masqué bloque toujours (défaillance du masquage).

Fermé par défaut : code de sortie 1 tant qu'un candidat bloquant est sans décision, ou que
des exemples à exclure sont encore présents (relancer select puis anonymize).

Sorties :
    data/interim/audit/revue_audit_noms.md    (relecture, NON versionné)
    reports/audit_noms.json                   (comptages)
    configs/exclusions_anonymisation.yaml     (identifiants exclus, cumulatif, lu par select.py)

Usage (depuis la racine du projet) :
    python -m src.data.audit_names
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterator

import spacy
import yaml

from src.data import reduce_chatdoctor as rcd
from src.data.anonymize import (CONFIG_PATH as ANON_CONFIG_PATH, DATASETS, OUTPUT_DIR as ANON_DIR,
                                TEXT_FIELDS, eponym_patterns, load_lexicon, load_private)
from src.data.sources import ROOT

DECISIONS_PATH = ROOT / "configs" / "decisions_audit_noms.yaml"
EXCLUSIONS_PATH = ROOT / "configs" / "exclusions_anonymisation.yaml"
REVIEW_PATH = ROOT / "data" / "interim" / "audit" / "revue_audit_noms.md"
REPORT_PATH = ROOT / "reports" / "audit_noms.json"

BATCH_SIZE = 256
CONTEXTS_PER_CANDIDATE = 2
LOWER_WORD = re.compile(r"(?<!\w)[a-zà-ÿ][a-zà-ÿ'’-]+")        # mot écrit en minuscules
OPTION_PREFIX = re.compile(r"^(?:[A-Ea-e][.)]\s+|[*•\-]+\s*)")   # "A. Cobalamin", "* Steatorrhea"
STRIP_CHARS = " *:.,;\"'()[]"
REAL_PEOPLE, FICTIONAL = "personnes_reelles", "cas_fictifs_et_editoriaux"
# Suffixes médicaux : maladies, actes, cellules, classes de médicaments (DCI)
MEDICAL_SUFFIX = re.compile(
    r"(?i)(?:itis|osis|oses|emia|aemia|uria|algia|pathy|plasty|ectomy|otomy|ostomy|scopy|graphy|"
    r"gram|oma|omas|cyte|cytes|blast|philia|penia|trophy|megaly|plegia|rrhea|rrhoea|cele|"
    r"mab|nib|pril|sartan|olol|azole|mycin|cillin|cycline|floxacin|statin|prazole|tidine|dipine|"
    r"azepam|zolam|triptan|gliptin|gliflozin|parin|xaban|lukast|setron|vudine|virine|"
    r"ite|ites|ose|émie|ectomie|otomie|scopie|graphie|pathie|plastie|algie|plégie|ase|ases)$")


def iter_texts() -> Iterator[tuple[str, str, str, str, str]]:
    """(jeu, identifiant, langue, périmètre, texte) pour chaque champ texte des 7 jeux anonymisés."""
    for name in DATASETS:
        with (ANON_DIR / f"{name}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                scope = REAL_PEOPLE if record.get("origine") == rcd.ORIGINE else FICTIONAL
                for field in TEXT_FIELDS:
                    text = record.get(field)
                    if isinstance(text, str) and text:
                        yield name, record["id"], record["langue"], scope, text


def clean_candidate(raw: str) -> str:
    return OPTION_PREFIX.sub("", raw.strip()).strip(STRIP_CHARS)


class CandidateFilter:
    """Filtres automatiques : retourne le motif d'écartement, ou None (décision humaine)."""

    def __init__(self, anon_cfg: dict[str, Any], decisions: dict[str, Any], vocabulary: set[str]) -> None:
        self.tags = {tag for tags in anon_cfg["remplacements"].values() for tag in tags.values()} | {rcd.PLACE_TAG}
        self.civilites = {c.rstrip(".").lower() for cs in anon_cfg["civilites"].values() for c in cs}
        self.before, self.after = eponym_patterns(anon_cfg["eponymes"])
        self.lexicon = {n.lower() for lang in anon_cfg["lexique_prenoms"] for n in load_lexicon(anon_cfg, lang)}
        self.non_noms = {str(x).lower() for x in decisions.get("non_noms") or []}
        self.exclusions = {str(x).lower() for x in decisions.get("exclure_si_contient") or []}
        self.vocabulary = vocabulary

    def reason(self, text: str, start: int, end: int) -> str | None:
        raw = text[start:end]
        candidate = clean_candidate(raw)
        low = candidate.lower()
        preceding, following = text[max(0, start - 40):start], text[end:end + 40]
        if any(tag in raw for tag in self.tags):
            return "balise"
        if len(candidate) < 3:
            return "forme"
        if low in self.non_noms:
            return "decision_non_nom"
        if low in self.exclusions:
            return "decision_exclusion"
        if low.rstrip(".") in self.civilites:
            return "civilite"
        if re.search(r"\d", candidate) or re.fullmatch(r"[A-Z.\-]{1,6}", candidate):
            return "sigle_ou_code"
        if candidate.endswith(("®", "™")) or following.startswith(("®", "™")):
            return "marque"
        if self.before.search(preceding) or self.after.search(following):
            return "eponyme"
        if re.match(r"\s*et\.?\s+al\b", following, re.IGNORECASE):
            return "auteur_cite"
        tokens = re.findall(r"[A-Za-zÀ-ÿ'’-]+", candidate)
        in_lexicon = any(t.lower() in self.lexicon for t in tokens)
        if tokens and not in_lexicon and all(t.lower() in self.vocabulary for t in tokens):
            return "vocabulaire_minuscule"
        if len(tokens) == 1 and not in_lexicon and MEDICAL_SUFFIX.search(tokens[0]):
            return "morphologie_medicale"
        return None

    def in_lexicon(self, candidate: str) -> bool:
        return any(t.lower() in self.lexicon for t in re.findall(r"[A-Za-zÀ-ÿ'’-]+", candidate))


def load_exclusions() -> dict[str, str]:
    if not EXCLUSIONS_PATH.exists():
        return {}
    data = yaml.safe_load(EXCLUSIONS_PATH.read_text(encoding="utf-8")) or {}
    return {e["id"]: e["raison"] for e in data.get("exemples_exclus") or []}


def write_exclusions(exclusions: dict[str, str]) -> None:
    """Fichier VERSIONNÉ : uniquement les identifiants (publics, ceux des datasets sources).
    Le terme qui a déclenché l'exclusion est un nom de personne réelle : il n'apparaît que
    dans la relecture locale, jamais dans ce fichier."""
    header = ["# Généré par src.data.audit_names — ne pas éditer à la main.",
              "# Exemples retirés par précaution (terme d'exclusion : configs/prive/, non versionné).",
              "# Cumulatif : un identifiant n'est jamais retiré de cette liste (sinon une sélection",
              "# ultérieure réintégrerait l'exemple). Lu par src.data.select : écartés de tous les jeux."]
    if exclusions:      # chaînes JSON entre guillemets : scalaires YAML valides
        body = ["exemples_exclus:"] + [
            f"  - {{id: {json.dumps(i, ensure_ascii=False)}, raison: \"précaution\"}}"
            for i in sorted(exclusions)]
    else:
        body = ["exemples_exclus: []"]
    EXCLUSIONS_PATH.write_text("\n".join(header + body) + "\n", encoding="utf-8")


def main() -> None:
    anon_cfg = yaml.safe_load(ANON_CONFIG_PATH.read_text(encoding="utf-8"))
    decisions = (yaml.safe_load(DECISIONS_PATH.read_text(encoding="utf-8")) or {}) if DECISIONS_PATH.exists() else {}
    # Termes d'exclusion privés (noms de personnes réelles), conservés hors du dépôt Git
    decisions["exclure_si_contient"] = (list(decisions.get("exclure_si_contient") or [])
                                        + list(load_private().get("exclure_si_contient") or []))
    bad = [x for k in ("non_noms", "exclure_si_contient") for x in decisions.get(k) or [] if not isinstance(x, str)]
    if bad:
        raise SystemExit(f"{DECISIONS_PATH.name} : valeurs non textuelles {bad} (guillemets manquants ?)")
    scope_mode = decisions.get("revue_exhaustive", "tout")
    if scope_mode not in ("tout", REAL_PEOPLE):
        raise SystemExit(f"revue_exhaustive : 'tout' ou '{REAL_PEOPLE}', pas {scope_mode!r}")

    print("Vocabulaire du corpus (mots écrits en minuscules)…")
    vocabulary = {w for *_, text in iter_texts() for w in LOWER_WORD.findall(text)}   # dernier champ = texte
    print(f"  {len(vocabulary)} mots")
    candidate_filter = CandidateFilter(anon_cfg, decisions, vocabulary)
    exclusion_terms = sorted(candidate_filter.exclusions, key=len, reverse=True)
    exclusion_re = (re.compile(rf"(?<!\w)(?:{'|'.join(map(re.escape, exclusion_terms))})(?!\w)", re.IGNORECASE)
                    if exclusion_terms else None)

    print("Chargement des modèles spaCy…")
    nlps = {lang: spacy.load(model, disable=["parser", "lemmatizer"])
            for lang, model in anon_cfg["modeles_spacy"].items()}

    stats: dict[str, Any] = {"textes": Counter(), "entites_per": 0, "filtres": Counter()}
    pending: dict[str, dict[str, Any]] = {}
    newly_excluded: dict[str, str] = {}
    buffers: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)

    def flush(lang: str) -> None:
        items, buffers[lang] = buffers[lang], []
        for (name, rid, scope, text), doc in zip(items, nlps[lang].pipe([t for *_, t in items], batch_size=64)):
            for ent in doc.ents:
                if ent.label_ not in ("PER", "PERSON"):
                    continue
                stats["entites_per"] += 1
                reason = candidate_filter.reason(text, ent.start_char, ent.end_char)
                if reason:
                    stats["filtres"][reason] += 1
                    continue
                candidate = clean_candidate(ent.text)
                entry = pending.setdefault(candidate.lower(), {"forme": candidate, "occurrences": 0,
                                                               "jeux": Counter(), "perimetres": Counter(),
                                                               "lexique": candidate_filter.in_lexicon(candidate),
                                                               "ids": set(), "contextes": []})
                entry["occurrences"] += 1
                entry["ids"].add(rid)
                entry["jeux"][name] += 1
                entry["perimetres"][scope] += 1
                if len(entry["contextes"]) < CONTEXTS_PER_CANDIDATE:
                    context = text[max(0, ent.start_char - 60):ent.end_char + 60].replace("\n", " ")
                    entry["contextes"].append(f"{name} `{rid}` : …{context}…")

    print("Analyse de tous les textes…")
    for count, (name, rid, lang, scope, text) in enumerate(iter_texts(), 1):
        stats["textes"][name] += 1
        stats.setdefault("textes_par_perimetre", Counter())[scope] += 1
        if exclusion_re and (m := exclusion_re.search(text)):
            newly_excluded.setdefault(rid, f"contient « {m.group(0)} »")
        buffers[lang].append((name, rid, scope, text))
        if len(buffers[lang]) >= BATCH_SIZE:
            flush(lang)
        if count % 5000 == 0:
            print(f"    {count} textes…")
    for lang in list(buffers):
        flush(lang)

    exclusions = {**load_exclusions(), **newly_excluded}       # cumulatif
    write_exclusions(exclusions)
    # Un candidat présent UNIQUEMENT dans des exemples exclus n'appelle pas de décision :
    # ces exemples disparaîtront ("Jean Duponts" quand "Jean Dupont" est exclu)
    covered = {k for k, e in pending.items() if e["ids"] <= exclusions.keys()}
    for key in covered:
        stats["filtres"]["exemple_exclu"] += pending.pop(key)["occurrences"]
    still_present = len(newly_excluded)                          # présents dans les jeux actuels

    def blocking(entry: dict[str, Any]) -> bool:
        """Bloque si le mode est strict, si le candidat figure dans un texte de personne
        réelle, ou s'il s'agit d'un prénom du lexique non masqué (défaillance du masquage)."""
        return scope_mode == "tout" or entry["perimetres"][REAL_PEOPLE] > 0 or entry["lexique"]

    blockers = {k: e for k, e in pending.items() if blocking(e)}
    by_scope = {REAL_PEOPLE: sum(1 for e in pending.values() if e["perimetres"][REAL_PEOPLE]),
                FICTIONAL: sum(1 for e in pending.values() if not e["perimetres"][REAL_PEOPLE]),
                "prenoms_du_lexique_non_masques": sum(1 for e in pending.values() if e["lexique"])}
    status = ("a_classer" if blockers else "exclusions_a_appliquer" if still_present else "valide")

    report = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "statut": status, "textes_audites": dict(stats["textes"]),
              "total_textes": sum(stats["textes"].values()), "entites_per": stats["entites_per"],
              "filtres_automatiques": dict(stats["filtres"].most_common()),
              "revue_exhaustive": scope_mode, "textes_par_perimetre": dict(stats["textes_par_perimetre"]),
              "candidats_total": len(pending), "candidats_par_perimetre": by_scope,
              "candidats_bloquants": len(blockers), "exemples_exclus_cumules": len(exclusions),
              "exemples_exclus_encore_presents": still_present}
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    review = ["# Contrôle exhaustif des noms — relecture", "",
              "Fichier de travail NON versionné : contient des extraits des données.", "",
              f"Statut : **{status}** — mode {scope_mode} — {report['total_textes']} textes, "
              f"{stats['entites_per']} entités PER, {len(blockers)} candidats bloquants "
              f"(sur {len(pending)}), {len(exclusions)} exemples exclus (cumul).", "",
              "Candidats par périmètre : " + ", ".join(f"{k} {v}" for k, v in by_scope.items()), "",
              "Filtres automatiques : " + ", ".join(f"{k} {v}" for k, v in stats["filtres"].most_common()), ""]
    sections = [("Prénoms du lexique non masqués (toujours bloquants)", lambda e: e["lexique"]),
                ("Personnes réelles — ChatDoctor", lambda e: not e["lexique"] and e["perimetres"][REAL_PEOPLE]),
                ("Cas fictifs et contenus éditoriaux" + (" (non bloquants en mode personnes_reelles)"
                                                         if scope_mode == REAL_PEOPLE else ""),
                 lambda e: not e["lexique"] and not e["perimetres"][REAL_PEOPLE])]
    for title, keep in sections:
        entries = sorted((e for e in pending.values() if keep(e)), key=lambda e: -e["occurrences"])
        review += [f"## {title} ({len(entries)}) — décision dans configs/{DECISIONS_PATH.name}", ""]
        for entry in entries:
            jeux = ", ".join(f"{k} {v}" for k, v in entry["jeux"].most_common())
            review.append(f"- **{entry['forme']}** (×{entry['occurrences']} ; {jeux})")
            review += [f"    - {c}" for c in entry["contextes"]]
        review.append("")
    review += ["", f"## Exemples exclus ({len(exclusions)})", ""]
    review += [f"- `{i}` : {r}" for i, r in sorted(exclusions.items())]
    REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_PATH.write_text("\n".join(review), encoding="utf-8")

    print(f"Statut : {status} — mode {scope_mode} — candidats bloquants : {len(blockers)} "
          f"(total {len(pending)} : {by_scope}) — exemples exclus : {len(exclusions)} "
          f"(dont {still_present} encore présents)")
    print(f"Relecture : {REVIEW_PATH}\nRapport : {REPORT_PATH}")
    if status == "a_classer":
        raise SystemExit(f"{len(blockers)} candidat(s) bloquant(s) sans décision : compléter {DECISIONS_PATH.name} "
                         "(ou les lexiques), relancer anonymize puis audit_names.")
    if status == "exclusions_a_appliquer":
        raise SystemExit(f"{still_present} exemple(s) à retirer : relancer select, anonymize, puis audit_names.")


if __name__ == "__main__":
    main()
