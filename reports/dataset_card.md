# CHSA — corpus médical bilingue pour SFT et DPO (POC triage)

> **Projet pédagogique** (OpenClassrooms, parcours AI Engineer) pour un client fictif, le Centre Hospitalier Saint-Aurélien (CHSA). Ce corpus et les modèles qui en sont issus ne constituent **pas** un dispositif médical et ne doivent pas servir à une décision clinique réelle.

Corpus médical bilingue (français / anglais) préparé pour le fine-tuning supervisé (SFT, avec LoRA) puis l'alignement par préférences (DPO) de **Qwen3-1.7B-Base**, dans le cadre d'un POC d'agent d'aide au triage aux urgences.

## Contenu

| Configuration | Split | Exemples |
|---|---|---|
| `sft` | `train` | 5000 |
| `sft` | `validation` | 500 |
| `sft` | `test` | 8821 |
| `sft` | `eval_clinique` | 263 |
| `dpo` | `train` | 3000 |
| `dpo` | `validation` | 300 |
| `dpo` | `test` | 777 |

```python
from datasets import load_dataset
sft = load_dataset("saidmo2/chsa-triage-medical", "sft")
dpo = load_dataset("saidmo2/chsa-triage-medical", "dpo")
```

Format conversationnel attendu par TRL : `messages` (system, user, assistant) pour le SFT ; `prompt` (system, user), `chosen` et `rejected` (assistant) pour le DPO. Le chat template (`Qwen/Qwen3-1.7B-Base`, mode réflexion désactivé) est appliqué par le tokenizer à l'entraînement comme à l'inférence.

Consigne système commune : *« Tu es l'assistant médical du service des urgences du Centre Hospitalier Saint-Aurélien (CHSA). Tu aides le personnel soignant et les patients en répondant avec exactitude aux questions médicales, dans la langue de la question. »*

## Sources

| Source | Dépôt Hugging Face | Révision épinglée | Licence | Rôle | Confiance |
|---|---|---|---|---|---|
| mediqal | `ANR-MALADES/MediQAl` | `5af34948a7` | Creative Commons Attribution 4.0 International (CC BY 4.0) | sft | élevée |
| frenchmedmcqa | `nthngdy/frenchmedmcqa` | `6195120803` | Apache-2.0 | sft | élevée |
| medquad | `keivalya/MedQuad-MedicalQnADataset` | `5b0961fbaa` | Creative Commons Attribution 4.0 International (CC BY 4.0) | sft | élevée |
| ultramedical_preference | `TsinghuaC3I/UltraMedical-Preference` | `761eb79353` | MIT | dpo | moyenne |

La licence de ce corpus dérivé (cc-by-4.0) est la plus restrictive des licences des sources ; références et modifications : voir la section « Citations et modifications ».

## Composition

- **SFT train** : 5000 paires instruction-réponse — sources : mediqal 3300, medquad 1400, frenchmedmcqa 300 ; tâches : mcq_single 2430, open_qa 1400, mcq_multi 1170 ; langues : fr 3600, en 1400.
- **DPO train** : 3000 paires de préférences (UltraMedical-Preference) — origines : ChatDoctor 1500, MedQA 750, MedQA-Evol 750 ; label_type : hard 2016, easy 984.
- **Évaluation** : tests officiels (QCM MediQAl et FrenchMedMCQA, 5 % de MedQuAD) ; `eval_clinique` = questions ouvertes MediQAl de médecine d'urgence et de réanimation, jamais utilisées à l'entraînement.

## Préparation

1. **Normalisation** vers un schéma commun, nettoyage tracé par exemple (`transformations`).
2. **Sélection** : priorité test > validation > train (aucun exemple d'évaluation dans l'entraînement, splits officiels dédoublonnés) ; filtres de spécialité et de longueur ; DPO limité aux paires `hard`/`easy` aux scores cohérents (paires `length` exclues) ; tirage stable par empreinte sha256.
3. **Anonymisation** (ci-dessous).
4. **Formatage** et contrôle des longueurs avec le chat template : SFT ≤ 1024 tokens ; DPO prompt ≤ 608, total ≤ 1536.

## Anonymisation (RGPD)

Principe : **des règles précises masquent** (civilités, lexiques de prénoms, motifs de contexte, coordonnées) ; **la NER spaCy, à fort rappel mais à faible précision mesurée sur ces textes, sert uniquement à l'audit**. Lieux et âges des cas fictifs sont conservés (information clinique) ; les éponymes médicaux sont protégés.

- Entités masquées : PERSON 2598, PHONE_NUMBER 41, URL 36, EMAIL_ADDRESS 2.
- **Personnes réelles** (paires ChatDoctor : 1723) : réduction structurelle (salutations, présentations, provenances, signatures supprimées), dates généralisées au mois, lieux masqués sur décision humaine, puis masquage — salutation_ouverture 918, date_generalisee 251, lieu_masque 245, signature 72, presentation 64, provenance 15, salutation_nominative 4.
- **Contrôle exhaustif** : 41399 textes analysés ; mode `personnes_reelles` — 100 % des textes de personnes réelles (5169) avec décision humaine pour chaque candidat ; 32 exemples retirés par précaution ; statut final : **valide**.
- Clés de dédoublonnage remplacées par leur empreinte sha256 ; les fichiers de relecture, qui contiennent les textes d'origine, restent hors du dépôt (non versionnés).

## Schéma

**SFT** : `id`, `messages`, `source`, `config`, `source_revision`, `langue`, `task` (mcq_single, mcq_multi, open_qa), `medical_subject`, `question_type`, `qtype`, `has_clinical_case`, `niveau_confiance`, `bloc`, `complement_de`, `symptomes`, `antecedents`, `constantes` (prévus pour des données de triage, non renseignés par ces sources), `transformations`, `entites_masquees`.

**DPO** : `id`, `prompt`, `chosen`, `rejected`, `origine`, `label_type`, `score_chosen`, `score_rejected`, `source_revision`, `langue`, `niveau_confiance`, `transformations`, `entites_masquees`, `reduction_chatdoctor` (JSON).

## Limites

- Les préférences DPO ont été jugées par GPT-4 (UltraMedical), **pas par des cliniciens**, et sont en anglais alors que l'agent cible un hôpital français.
- Aucun exemple de la tâche de triage elle-même : le bloc prévu à cet effet a été complété par des QCM MediQAl (`complement_de`).
- Cas d'examen et contenus éditoriaux (personnages fictifs) : anonymisation automatique, mesurée par échantillons (précision d'environ 97 %), sans relecture exhaustive.
- Bruit résiduel des sources (fautes de saisie, artefacts d'encodage) non corrigé.

## Citations et modifications

Ce corpus est dérivé des sources suivantes, qu'il convient de citer :

- **mediqal** — Bazoge A. (2025). MediQAl: A French Medical Question Answering Dataset for Knowledge and Reasoning Evaluation. arXiv:2507.20917.
- **frenchmedmcqa** — Labrak Y., Bazoge A., Dufour R., Daille B., Gourraud P.-A., Morin E., Rouvier M. (2022). FrenchMedMCQA: A French Multiple-Choice Question Answering Dataset for Medical domain. Workshop LOUHI 2022.
- **medquad** — Ben Abacha A., Demner-Fushman D. (2019). A question-entailment approach to question answering. BMC Bioinformatics 20, 511.
- **ultramedical_preference** — Zhang K. et al. (2024). UltraMedical: Building Specialized Generalists in Biomedicine. TsinghuaC3I.

**Modifications apportées** (exigence CC BY 4.0) : normalisation et nettoyage du texte, reformulation des QCM en paires instruction-réponse, sélection d'un sous-ensemble, anonymisation (masquage, réduction structurelle des messages ChatDoctor, généralisation des dates), retrait d'exemples par précaution, mise au format conversationnel. Les textes conservés ne sont pas réécrits sur le fond.

## Reproductibilité

Pipeline : `normalize` → `select` → `anonymize` → `audit_names` → `build_dataset` → `publish_dataset` (dossier `src/data/`, paramètres dans `configs/`, rapports dans `reports/`).

*Carte générée le 2026-10-04.*
