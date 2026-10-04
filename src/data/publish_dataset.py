"""Publication du dataset sur le Hugging Face Hub (livrable 1) et carte de dataset.

Un seul dépôt, deux configurations :
  - sft : train, validation, test, eval_clinique
  - dpo : train, validation, test
La carte (README.md) est générée à partir des rapports du pipeline : sources, licences,
révisions, composition, schéma, anonymisation chiffrée, limites. Elle ne contient que des
agrégats : une copie est versionnée dans reports/dataset_card.md.

Contrôles (fermés par défaut) : audit des noms « valide » et formatage « succes ».

Usage (depuis la racine du projet, après « hf auth login ») :
    python -m src.data.publish_dataset                    # publie (privé à la création)
    python -m src.data.publish_dataset --carte-seule      # génère la carte, sans publier
    python -m src.data.publish_dataset --visibilite prive|public   # change la visibilité seulement
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import yaml

from src.data.sources import ROOT, load_registry

CONFIG_PATH = ROOT / "configs" / "publication.yaml"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
CARD_COPY_PATH = REPORTS_DIR / "dataset_card.md"

CONFIANCE = {"eleve": "élevée", "moyen": "moyenne", "faible": "faible"}

# Références des sources (CC BY 4.0 : attribution obligatoire) — à vérifier sur la carte de chaque dataset
CITATIONS = {
    "mediqal": "Bazoge A. (2025). MediQAl: A French Medical Question Answering Dataset for Knowledge and "
               "Reasoning Evaluation. arXiv:2507.20917.",
    "frenchmedmcqa": "Labrak Y., Bazoge A., Dufour R., Daille B., Gourraud P.-A., Morin E., Rouvier M. (2022). "
                     "FrenchMedMCQA: A French Multiple-Choice Question Answering Dataset for Medical domain. "
                     "Workshop LOUHI 2022.",
    "medquad": "Ben Abacha A., Demner-Fushman D. (2019). A question-entailment approach to question answering. "
               "BMC Bioinformatics 20, 511.",
    "ultramedical_preference": "Zhang K. et al. (2024). UltraMedical: Building Specialized Generalists in "
                               "Biomedicine. TsinghuaC3I.",
}

SPLITS = {"sft": {"train": "sft_train", "validation": "sft_validation", "test": "sft_test",
                  "eval_clinique": "eval_clinique"},
          "dpo": {"train": "dpo_train", "validation": "dpo_validation", "test": "dpo_test"}}


def read_report(name: str) -> dict[str, Any]:
    path = REPORTS_DIR / name
    if not path.exists():
        raise SystemExit(f"Rapport manquant : {path} (lancer le pipeline complet).")
    return json.loads(path.read_text(encoding="utf-8"))


def check_gates() -> None:
    audit, fmt = read_report("audit_noms.json"), read_report("format.json")
    if audit.get("statut") != "valide":
        raise SystemExit(f"Audit des noms au statut « {audit.get('statut')} » : publication refusée.")
    if fmt.get("statut") != "succes":
        raise SystemExit(f"Formatage au statut « {fmt.get('statut')} » : publication refusée.")


def count_lines(name: str) -> int:
    with (PROCESSED_DIR / f"{name}.jsonl").open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def fmt_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------- carte de dataset
def build_card(cfg: dict[str, Any], repo_id: str) -> str:
    selection, anonym = read_report("selection.json"), read_report("anonymisation.json")
    audit, fmt = read_report("audit_noms.json"), read_report("format.json")
    registry = load_registry()
    contenu = selection.get("contenu", {})
    jeux_anonym = anonym.get("jeux", {})

    entites, reduction, paires_cd = Counter(), Counter(), 0
    for stats in jeux_anonym.values():
        entites.update(stats.get("entites_masquees", {}))
        reduction.update(stats.get("reduction_chatdoctor", {}))
        paires_cd += stats.get("paires_chatdoctor", 0)
    limites = fmt.get("limites_entrainement", {})

    lines = [
        f"# {cfg['pretty_name']}", "",
        "> **Projet pédagogique** (OpenClassrooms, parcours AI Engineer) pour un client fictif, le Centre "
        "Hospitalier Saint-Aurélien (CHSA). Ce corpus et les modèles qui en sont issus ne constituent **pas** "
        "un dispositif médical et ne doivent pas servir à une décision clinique réelle.", "",
        "Corpus médical bilingue (français / anglais) préparé pour le fine-tuning supervisé (SFT, avec LoRA) "
        "puis l'alignement par préférences (DPO) de **Qwen3-1.7B-Base**, dans le cadre d'un POC d'agent "
        "d'aide au triage aux urgences.", "",
        "## Contenu", "",
        "| Configuration | Split | Exemples |", "|---|---|---|",
    ]
    for config, splits in SPLITS.items():
        for split, name in splits.items():
            lines.append(f"| `{config}` | `{split}` | {count_lines(name)} |")
    lines += ["", "```python", "from datasets import load_dataset",
              f'sft = load_dataset("{repo_id}", "sft")', f'dpo = load_dataset("{repo_id}", "dpo")', "```", "",
              "Format conversationnel attendu par TRL : `messages` (system, user, assistant) pour le SFT ; "
              "`prompt` (system, user), `chosen` et `rejected` (assistant) pour le DPO. Le chat template "
              f"(`{fmt.get('chat_template')}`, mode réflexion désactivé) est appliqué par le tokenizer à "
              "l'entraînement comme à l'inférence.", "",
              f"Consigne système commune : *« {fmt.get('system_prompt')} »*", ""]

    lines += ["## Sources", "",
              "| Source | Dépôt Hugging Face | Révision épinglée | Licence | Rôle | Confiance |",
              "|---|---|---|---|---|---|"]
    for name, src in registry.items():
        confiance = CONFIANCE.get(src.get("niveau_confiance"), src.get("niveau_confiance"))
        lines.append(f"| {name} | `{src['hf_id']}` | `{str(src.get('revision', ''))[:10]}` | {src.get('licence')} "
                     f"| {src.get('role_prevu')} | {confiance} |")
    lines += ["", f"La licence de ce corpus dérivé ({cfg['licence']}) est la plus restrictive des licences des "
              "sources ; références et modifications : voir la section « Citations et modifications ».", ""]

    sft_train, dpo_train = contenu.get("sft_train", {}), contenu.get("dpo_train", {})
    lines += ["## Composition", "",
              f"- **SFT train** : {sft_train.get('total')} paires instruction-réponse — sources : "
              f"{fmt_counts(sft_train.get('source', {}))} ; tâches : {fmt_counts(sft_train.get('task', {}))} ; "
              f"langues : {fmt_counts(sft_train.get('langue', {}))}.",
              f"- **DPO train** : {dpo_train.get('total')} paires de préférences (UltraMedical-Preference) — "
              f"origines : {fmt_counts(dpo_train.get('origine', {}))} ; label_type : "
              f"{fmt_counts(dpo_train.get('label_type', {}))}.",
              "- **Évaluation** : tests officiels (QCM MediQAl et FrenchMedMCQA, 5 % de MedQuAD) ; "
              "`eval_clinique` = questions ouvertes MediQAl de médecine d'urgence et de réanimation, "
              "jamais utilisées à l'entraînement.", ""]

    lines += ["## Préparation", "",
              "1. **Normalisation** vers un schéma commun, nettoyage tracé par exemple (`transformations`).",
              "2. **Sélection** : priorité test > validation > train (aucun exemple d'évaluation dans "
              "l'entraînement, splits officiels dédoublonnés) ; filtres de spécialité et de longueur ; DPO limité "
              "aux paires `hard`/`easy` aux scores cohérents (paires `length` exclues) ; tirage stable par "
              "empreinte sha256.",
              "3. **Anonymisation** (ci-dessous).",
              f"4. **Formatage** et contrôle des longueurs avec le chat template : SFT ≤ "
              f"{limites.get('sft_max_seq_length')} tokens ; DPO prompt ≤ {limites.get('dpo_max_prompt_length')}, "
              f"total ≤ {limites.get('dpo_max_length')}.", ""]

    perim = audit.get("textes_par_perimetre", {})
    lines += ["## Anonymisation (RGPD)", "",
              "Principe : **des règles précises masquent** (civilités, lexiques de prénoms, motifs de contexte, "
              "coordonnées) ; **la NER spaCy, à fort rappel mais à faible précision mesurée sur ces textes, sert "
              "uniquement à l'audit**. Lieux et âges des cas fictifs sont conservés (information clinique) ; "
              "les éponymes médicaux sont protégés.", "",
              f"- Entités masquées : {fmt_counts(dict(entites))}.",
              f"- **Personnes réelles** (paires ChatDoctor : {paires_cd}) : réduction structurelle (salutations, "
              "présentations, provenances, signatures supprimées), dates généralisées au mois, lieux masqués sur "
              f"décision humaine, puis masquage — {fmt_counts(dict(reduction))}.",
              f"- **Contrôle exhaustif** : {audit.get('total_textes')} textes analysés ; mode "
              f"`{audit.get('revue_exhaustive')}` — 100 % des textes de personnes réelles "
              f"({perim.get('personnes_reelles')}) avec décision humaine pour chaque candidat ; "
              f"{audit.get('exemples_exclus_cumules')} exemples retirés par précaution ; statut final : "
              f"**{audit.get('statut')}**.",
              "- Clés de dédoublonnage remplacées par leur empreinte sha256 ; les fichiers de relecture, qui "
              "contiennent les textes d'origine, restent hors du dépôt (non versionnés).", ""]

    lines += ["## Schéma", "",
              "**SFT** : `id`, `messages`, `source`, `config`, `source_revision`, `langue`, `task` "
              "(mcq_single, mcq_multi, open_qa), `medical_subject`, `question_type`, `qtype`, `has_clinical_case`, "
              "`niveau_confiance`, `bloc`, `complement_de`, `symptomes`, `antecedents`, `constantes` (prévus pour "
              "des données de triage, non renseignés par ces sources), `transformations`, `entites_masquees`.", "",
              "**DPO** : `id`, `prompt`, `chosen`, `rejected`, `origine`, `label_type`, `score_chosen`, "
              "`score_rejected`, `source_revision`, `langue`, `niveau_confiance`, `transformations`, "
              "`entites_masquees`, `reduction_chatdoctor` (JSON).", ""]

    lines += ["## Limites", "",
              "- Les préférences DPO ont été jugées par GPT-4 (UltraMedical), **pas par des cliniciens**, et sont "
              "en anglais alors que l'agent cible un hôpital français.",
              "- Aucun exemple de la tâche de triage elle-même : le bloc prévu à cet effet a été complété par des "
              "QCM MediQAl (`complement_de`).",
              "- Cas d'examen et contenus éditoriaux (personnages fictifs) : anonymisation automatique, mesurée "
              "par échantillons (précision d'environ 97 %), sans relecture exhaustive.",
              "- Bruit résiduel des sources (fautes de saisie, artefacts d'encodage) non corrigé.", ""]

    lines += ["## Citations et modifications", "",
              "Ce corpus est dérivé des sources suivantes, qu'il convient de citer :", ""]
    lines += [f"- **{name}** — {CITATIONS.get(name, src['hf_id'])}" for name, src in registry.items()]
    lines += ["", "**Modifications apportées** (exigence CC BY 4.0) : normalisation et nettoyage du texte, "
              "reformulation des QCM en paires instruction-réponse, sélection d'un sous-ensemble, anonymisation "
              "(masquage, réduction structurelle des messages ChatDoctor, généralisation des dates), retrait "
              "d'exemples par précaution, mise au format conversationnel. Les textes conservés ne sont pas "
              "réécrits sur le fond.", ""]
    lines += ["## Reproductibilité", "",
              "Pipeline : `normalize` → `select` → `anonymize` → `audit_names` → `build_dataset` → "
              "`publish_dataset` (dossier `src/data/`, paramètres dans `configs/`, rapports dans `reports/`).", "",
              f"*Carte générée le {datetime.now(timezone.utc).strftime('%Y-%m-%d')}.*"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- Hub
def set_visibility(api, repo_id: str, private: bool) -> None:
    if hasattr(api, "update_repo_settings"):
        api.update_repo_settings(repo_id=repo_id, private=private, repo_type="dataset")
    else:                                   # anciennes versions de huggingface_hub
        api.update_repo_visibility(repo_id=repo_id, private=private, repo_type="dataset")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--carte-seule", action="store_true", help="générer la carte sans publier")
    group.add_argument("--visibilite", choices=["prive", "public"], help="changer seulement la visibilité")
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    from huggingface_hub import DatasetCard, HfApi
    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception:                       # pas de session : acceptable seulement pour --carte-seule
        if not args.carte_seule:
            raise SystemExit("Pas de session Hugging Face : lancer « hf auth login ».")
        user = "<utilisateur>"
    repo_id = f"{user}/{cfg['repo_name']}"

    if args.visibilite:
        set_visibility(api, repo_id, private=args.visibilite == "prive")
        print(f"{repo_id} : visibilité « {args.visibilite} ».")
        if args.visibilite == "public":
            print("Attention : toute copie faite pendant la période publique subsistera après un retour en privé.")
        return

    check_gates()
    card = build_card(cfg, repo_id)
    CARD_COPY_PATH.write_text(card, encoding="utf-8")
    (PROCESSED_DIR / "README.md").write_text(card, encoding="utf-8")
    print(f"Carte : {CARD_COPY_PATH}")
    if args.carte_seule:
        return

    from datasets import load_dataset
    for config, splits in SPLITS.items():
        files = {split: str(PROCESSED_DIR / f"{name}.jsonl") for split, name in splits.items()}
        dataset = load_dataset("json", data_files=files)
        print(f"[{config}] envoi : {', '.join(f'{s} {len(d)}' for s, d in dataset.items())}")
        dataset.push_to_hub(repo_id, config_name=config, private=cfg["private"])

    # push_to_hub a écrit dans le README les métadonnées des configurations : on les conserve
    # et on remplace seulement le texte et les champs descriptifs.
    hub_card = DatasetCard.load(repo_id)
    hub_card.data.license = cfg["licence"]
    hub_card.data.language = cfg["langues"]
    hub_card.data.pretty_name = cfg["pretty_name"]
    hub_card.data.task_categories = cfg["task_categories"]
    hub_card.data.tags = cfg["tags"]
    hub_card.text = card
    hub_card.push_to_hub(repo_id)
    print(f"Publié : https://huggingface.co/datasets/{repo_id} (privé à la création : {cfg['private']})")


if __name__ == "__main__":
    main()
